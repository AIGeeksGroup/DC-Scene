"""Frozen model scoring with full training coverage and auditable provenance."""
import torch
import torch.nn.functional as F

from .data import loader, to_device
from .quality import atomic_json, fingerprint, pair_id


@torch.no_grad()
def score_pairs(model, dataset, args, output, alignment_model=None, provenance=None):
    model.eval()
    if alignment_model is not None:
        alignment_model.eval()
    device = next(model.parameters()).device
    rows = []
    for step, batch in enumerate(loader(dataset, args)):
        batch = to_device(batch, device)
        prediction = model(batch, is_eval=False)  # teacher forcing, eval-mode dropout/BN
        if not torch.all(prediction["caption_count"] == 1):
            raise ValueError("Every scored pair needs exactly one matched caption")
        alignment = alignment_model(batch, is_eval=False) if alignment_model is not None else prediction
        if "scene_embeddings" not in alignment:
            raise ValueError("Vote2Cap-DETR++ needs a pretrained 3D CoCa alignment model or imported scores")
        # CoCa's trained joint latent space is used; no random 3D-to-CLIP projection.
        similarity = (F.normalize(alignment["scene_embeddings"].float(), dim=-1) *
                      F.normalize(alignment["text_embeddings"].float(), dim=-1)).sum(-1)
        for index, score, loss in zip(batch["pair_index"].tolist(), similarity.tolist(), prediction["caption_nll"].tolist()):
            record = dataset.records[index]
            rows.append(dict(pair_id=pair_id(record), scene_id=record["scene_id"],
                             object_id=record["object_id"], ann_id=record["ann_id"],
                             clip_score=score, caption_nll=loss))
        if step % args.log_every == 0:
            print(f"Scored {len(rows)}/{len(dataset)} training pairs", flush=True)
    payload = dict(version=1, split="train", dataset_fingerprint=fingerprint(dataset.records),
                   provenance=provenance or {}, alignment="pretrained_coca_joint_latents_cosine",
                   loss_reduction="token_sum_including_eos", pairs=rows)
    # Validate coverage and finiteness before publishing the table.
    from .quality import QualityTable
    QualityTable(payload, dataset.records)
    atomic_json(output, payload)
    return payload
