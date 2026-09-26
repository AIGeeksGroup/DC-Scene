import math
import time
from pathlib import Path

import torch

from .checkpoint import load_weights, restore_rng, save
from .data import loader, to_device
from .quality import Curriculum, QualityTable, atomic_json, fingerprint, pair_id


def make_curriculum(args, records):
    ids = [pair_id(row) for row in records]
    if args.strategy in ("full", "random"):
        import numpy as np
        ranked = np.random.default_rng(args.seed).permutation(len(records)).tolist()
        score_hash, bounds = None, None
    else:
        if not args.quality_scores:
            raise ValueError("Generate --quality_scores before quality-guided training")
        table = QualityTable.load(args.quality_scores, records)
        ranked = table.rank(args.strategy, args.lower_quantile, args.upper_quantile,
                            args.alignment_weight, args.seed, args.top_k_per_scene)
        score_hash, bounds = table.digest, table.bounds
    ratios = [1.0] if args.strategy == "full" else args.ratios
    epochs = [sum(args.stage_epochs)] if args.strategy == "full" else args.stage_epochs
    curriculum = Curriculum(ranked, len(records), ratios, epochs)
    manifest = curriculum.manifest(ids)
    manifest.update(dataset_fingerprint=fingerprint(records), quality_fingerprint=score_hash,
                    strategy=args.strategy, bounds=bounds, seed=args.seed)
    return curriculum, manifest


def train(args, model, dataset_config, dataset, validation):
    curriculum, manifest = make_curriculum(args, dataset.records)
    manifest["training"] = {key: getattr(args, key) for key in (
        "backbone", "dataset", "num_points", "max_des_len", "use_color", "use_normal",
        "use_height", "use_multiview", "base_lr", "final_lr", "weight_decay",
        "batchsize_per_gpu", "clip_gradient", "dataset_num_workers", "refresh_at_stages",
        "lower_quantile", "upper_quantile", "alignment_weight", "top_k_per_scene",
        "test_min_iou", "eval_every_epochs", "use_beam_search")}
    if args.refresh_at_stages and args.alignment_checkpoint:
        from .checkpoint import file_digest
        manifest["alignment_checkpoint_sha256"] = file_digest(args.alignment_checkpoint)
    # Initial configuration stays immutable; refreshed rankings are runtime state.
    contract = fingerprint(manifest)
    manifest["contract"] = contract
    directory = Path(args.checkpoint_dir)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.base_lr, weight_decay=args.weight_decay)
    start, step, samples, best = 0, 0, 0, -math.inf
    if args.resume:
        checkpoint = load_weights(model, args.resume)
        if checkpoint["manifest"].get("contract") != contract:
            raise ValueError("Resume requires identical data, scores, curriculum, and training settings")
        manifest = checkpoint["manifest"]
        by_id = {pair_id(record): i for i, record in enumerate(dataset.records)}
        curriculum.ranked = [by_id[key] for key in manifest["ranked_pair_ids"]]
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint)
        start, step, samples, best = checkpoint["epoch"] + 1, checkpoint["step"], checkpoint["samples"], checkpoint["best"]
    atomic_json(directory / "curriculum.json", manifest)
    device = next(model.parameters()).device
    total_epochs = curriculum.ends[-1]
    for epoch in range(start, total_epochs):
        if args.refresh_at_stages and epoch in curriculum.ends[:-1]:
            import copy
            import numpy as np
            import random
            from .data import PairDataset
            from .scoring import score_pairs
            from .checkpoint import file_digest
            from models.model_general import CaptionNet
            # Scoring is an eval/no-grad pass and does not update the optimizer.
            states = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
            score_dataset = PairDataset(dataset.scenes, seed=args.seed, deterministic=True)
            dataset.scenes.augment = False
            alignment = None
            try:
                if args.backbone == "vote2cap_detrpp":
                    alignment_args = copy.copy(args)
                    alignment_args.backbone = "3dcoca"
                    alignment = CaptionNet(alignment_args, dataset_config, dataset).to(device)
                    load_weights(alignment, args.alignment_checkpoint)
                score_path = directory / f"quality_epoch{epoch}.json"
                provenance = {"epoch": epoch, "global_step": step,
                              "source_checkpoint_sha256": file_digest(directory / "checkpoint_last.pth" if (directory / "checkpoint_last.pth").exists() else args.resume)}
                score_pairs(model, score_dataset, args, score_path, alignment, provenance)
                table = QualityTable.load(score_path, dataset.records)
                ranked = table.rank("diq", args.lower_quantile, args.upper_quantile,
                                    args.alignment_weight, args.seed, args.top_k_per_scene)
                # Refresh can change membership. Ratios still refer to original N.
                curriculum = Curriculum(ranked, len(dataset), args.ratios, args.stage_epochs)
                manifest["ranked_pair_ids"] = [pair_id(dataset.records[i]) for i in ranked]
                manifest["refreshed_quality_fingerprint"] = table.digest
                manifest["bounds"] = table.bounds
                atomic_json(directory / "curriculum.json", manifest)
            finally:
                dataset.scenes.augment = True
                random.setstate(states[0])
                np.random.set_state(states[1])
                torch.set_rng_state(states[2])
                torch.cuda.set_rng_state_all(states[3])
                del alignment
        indices = curriculum.indices(epoch)
        train_loader = loader(torch.utils.data.Subset(dataset, indices), args, shuffle=True, epoch=epoch)
        model.train()
        started, loss_sum, count = time.monotonic(), 0.0, 0
        print(f"Epoch {epoch + 1}/{total_epochs}, stage {curriculum.stage(epoch) + 1}, "
              f"pairs {len(indices)}/{len(dataset)}", flush=True)
        for batch_index, batch in enumerate(train_loader):
            # Epoch-based decay stays continuous when the dataloader changes length.
            progress = (epoch + batch_index / len(train_loader)) / total_epochs
            lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, is_eval=False)
            loss = output["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, step {step}")
            loss.backward()
            if args.clip_gradient > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()
            n = batch["point_clouds"].shape[0]
            loss_sum += float(loss.detach()) * n
            count += n
            samples += n
            step += 1
            if step % args.log_every == 0:
                print(f"step={step} loss={float(loss.detach()):.4f} lr={lr:.3g} processed_pairs={samples}", flush=True)
        metrics = {}
        improved = False
        if validation is not None and ((epoch + 1) % args.eval_every_epochs == 0 or epoch + 1 == total_epochs):
            from evaluation import evaluate_caption
            metrics = evaluate_caption(args, epoch, model, dataset_config, loader(validation, args, epoch=epoch))
            score = metrics[f"CiDEr@{args.test_min_iou}"]
            if score > best:
                best, improved = score, True
        report = dict(epoch=epoch + 1, stage=curriculum.stage(epoch) + 1, pairs=count,
                      processed_pairs=samples, loss=loss_sum / count,
                      seconds=time.monotonic() - started, metrics=metrics)
        import json
        with (directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
        save(directory / "checkpoint_last.pth", model, optimizer, epoch, step, samples, best, manifest, args)
        if improved:
            save(directory / "checkpoint_best.pth", model, optimizer, epoch, step, samples, best, manifest, args)
        if epoch + 1 in curriculum.ends:
            save(directory / f"checkpoint_stage{curriculum.stage(epoch) + 1}.pth",
                 model, optimizer, epoch, step, samples, best, manifest, args)
