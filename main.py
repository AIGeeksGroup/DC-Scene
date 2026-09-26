"""DC-Scene: score, train, evaluate, or export dense scene captions."""
import argparse
import json
import os
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=["score", "train", "evaluate", "predict"], default="train")
    parser.add_argument("--backbone", choices=["3dcoca", "vote2cap_detrpp"], default="3dcoca")
    parser.add_argument("--dataset", choices=["scene_scanrefer", "scene_nr3d"], default="scene_scanrefer")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--scan_root", default="data/scannet/scannet_data")
    parser.add_argument("--metadata_root", default="data/scannet/meta_data")
    parser.add_argument("--checkpoint_dir", default="outputs/dc_scene")
    parser.add_argument("--checkpoint", help="Strictly compatible pretrained weights for scoring/evaluation")
    parser.add_argument("--init", help="Training warm start; use --allow_partial_init for upstream migration")
    parser.add_argument("--allow_partial_init", action="store_true")
    parser.add_argument("--resume", help="DC-Scene epoch checkpoint including optimizer and curriculum")
    parser.add_argument("--alignment_checkpoint", help="Strict pretrained CoCa scorer for Vote2Cap scores")
    parser.add_argument("--quality_scores", help="Train-only JSON quality table")
    parser.add_argument("--output", default="outputs/quality.json")
    parser.add_argument("--point_cloud", help="Prediction .npy array with xyz,rgb,normal columns")
    parser.add_argument("--multiview_features", help="Prediction .npy array [N,128]")
    parser.add_argument("--strategy", choices=["diq", "clip", "loss", "random", "full"], default="diq")
    parser.add_argument("--ratios", type=float, nargs="+", default=[.25, .5, .75])
    parser.add_argument("--stage_epochs", type=int, nargs="+", default=[360, 360, 360])
    parser.add_argument("--lower_quantile", type=float, default=.05)
    parser.add_argument("--upper_quantile", type=float, default=.95)
    parser.add_argument("--alignment_weight", type=float, default=.5)
    parser.add_argument("--top_k_per_scene", type=int)
    parser.add_argument("--refresh_at_stages", action="store_true", help="Rescore all training pairs at stage transitions")
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--final_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=.1)
    parser.add_argument("--clip_gradient", type=float, default=.1)
    parser.add_argument("--batchsize_per_gpu", type=int, default=8)
    parser.add_argument("--dataset_num_workers", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=40000)
    parser.add_argument("--max_des_len", type=int, default=32)
    parser.add_argument("--use_color", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_normal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_multiview", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no_height", action="store_true")
    parser.add_argument("--use_beam_search", action="store_true")
    parser.add_argument("--eval_every_epochs", type=int, default=20, help="0 disables validation during training")
    parser.add_argument("--test_min_iou", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--log_every", type=int, default=20)
    known, _ = parser.parse_known_args()
    if known.config:
        settings = json.loads(known.config.read_text())
        valid = {action.dest for action in parser._actions} - {"help", "config"}
        if settings.keys() - valid:
            parser.error(f"Unknown config keys: {settings.keys() - valid}")
        parser.set_defaults(**settings)
    args = parser.parse_args()
    args.use_height, args.vocabulary = not args.no_height, "scanrefer"
    args.max_epoch = sum(args.stage_epochs)
    if args.init and args.resume:
        parser.error("Choose --init or --resume")
    if args.mode != "train" and (args.init or args.resume):
        parser.error("--init and --resume apply only to training")
    if args.allow_partial_init and not args.init:
        parser.error("--allow_partial_init requires --init")
    if args.mode == "train" and args.checkpoint:
        parser.error("Use --init or --resume for training")
    if args.mode in ("score", "evaluate", "predict") and not args.checkpoint:
        parser.error("This mode requires --checkpoint")
    if args.mode == "predict" and not args.point_cloud:
        parser.error("Prediction requires --point_cloud")
    if args.backbone == "3dcoca" and args.use_beam_search:
        parser.error("CoCa uses autoregressive greedy decoding; beam search is supported for Vote2Cap-DETR++")
    if args.backbone == "3dcoca" and not (args.use_color or args.use_normal or args.use_multiview or args.use_height):
        parser.error("The sparse CoCa encoder requires at least one input feature in addition to XYZ")
    if args.batchsize_per_gpu < 1 or args.num_points < 2048 or args.max_des_len < 2 or args.log_every < 1 or args.eval_every_epochs < 0:
        parser.error("Invalid batch size, point count, caption length, or logging/evaluation interval")
    if args.mode == "score" and args.backbone == "vote2cap_detrpp" and not args.alignment_checkpoint:
        parser.error("Vote2Cap scoring requires --alignment_checkpoint (pretrained 3D CoCa)")
    if args.refresh_at_stages and args.strategy != "diq":
        parser.error("Stage refresh is defined for --strategy diq")
    if args.refresh_at_stages and args.backbone == "vote2cap_detrpp" and not args.alignment_checkpoint:
        parser.error("Vote2Cap stage refresh needs --alignment_checkpoint")
    args.config = str(args.config) if args.config else None
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["DC_SCENE_DATA"] = str(Path(args.data_root).resolve())
    os.environ["DC_SCENE_SCANS"] = str(Path(args.scan_root).resolve())
    os.environ["DC_SCENE_METADATA"] = str(Path(args.metadata_root).resolve())
    directory = Path(args.checkpoint_dir)
    if args.mode == "train" and not args.resume and directory.exists() and any(directory.glob("checkpoint*.pth")):
        raise ValueError("Training directory already contains checkpoints; use --resume or a new directory")
    directory.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from dc_scene.checkpoint import file_digest, load_weights
    from dc_scene.data import build_datasets, loader
    from dc_scene.quality import atomic_json
    from models.model_general import CaptionNet
    if not torch.cuda.is_available():
        raise RuntimeError("The reference PointNet++ backbones require an NVIDIA CUDA GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.mode == "predict":
        from dc_scene.inference import inference_metadata
        config, pairs = inference_metadata(args)
        validation = None
    else:
        config, pairs, validation = build_datasets(args, scoring=args.mode != "train",
            evaluation=args.mode == "evaluate" or (args.mode == "train" and args.eval_every_epochs > 0))
    model = CaptionNet(args, config, pairs).cuda()
    if args.checkpoint:
        load_weights(model, args.checkpoint)
    if args.init:
        load_weights(model, args.init, allow_partial=args.allow_partial_init)
    if args.mode == "score":
        from dc_scene.scoring import score_pairs
        alignment = None
        provenance = dict(checkpoint_sha256=file_digest(args.checkpoint), backbone=args.backbone,
                          seed=args.seed, num_points=args.num_points, max_des_len=args.max_des_len)
        if args.alignment_checkpoint:
            import copy
            alignment_args = copy.copy(args)
            alignment_args.backbone = "3dcoca"
            alignment = CaptionNet(alignment_args, config, pairs).cuda()
            load_weights(alignment, args.alignment_checkpoint)
            provenance["alignment_checkpoint_sha256"] = file_digest(args.alignment_checkpoint)
        score_pairs(model, pairs, args, args.output, alignment, provenance)
    elif args.mode == "train":
        from dc_scene.training import train
        atomic_json(directory / "config.json", vars(args))
        train(args, model, config, pairs, validation)
    elif args.mode == "evaluate":
        from evaluation import evaluate_caption
        metrics = evaluate_caption(args, -1, model, config, loader(validation, args))
        atomic_json(directory / "metrics.json", metrics)
    else:
        from dc_scene.inference import predict
        predict(args, model)


if __name__ == "__main__":
    main()
