"""Import exact external scene/text embeddings and caption NLL for DIQ scoring.

NPZ keys: pair_ids [N] (Unicode strings), scene_embeddings [N,D],
text_embeddings [N,D], caption_nll [N]. Both embeddings must already share
a pretrained, aligned latent space. NLL is the masked sum, not perplexity.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dc_scene.checkpoint import file_digest
from dc_scene.quality import QualityTable, atomic_json, fingerprint, pair_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder_description", required=True, help="Encoder/checkpoint and preprocessing provenance")
    args = parser.parse_args()
    records = json.loads(args.annotations.read_text())
    with np.load(args.features, allow_pickle=False) as cache:
        ids = cache["pair_ids"].tolist()
        scene = cache["scene_embeddings"].astype(np.float64)
        text = cache["text_embeddings"].astype(np.float64)
        losses = cache["caption_nll"]
    n = len(records)
    if scene.ndim != 2 or scene.shape != text.shape or len(scene) != n or losses.shape != (n,) or len(ids) != n:
        raise ValueError("Invalid embedding/NLL dimensions")
    if not np.isfinite(scene).all() or not np.isfinite(text).all():
        raise ValueError("Nonfinite embeddings")
    sn, tn = np.linalg.norm(scene, axis=-1), np.linalg.norm(text, axis=-1)
    if (sn == 0).any() or (tn == 0).any():
        raise ValueError("Zero-norm embeddings cannot define alignment")
    scores = (scene * text).sum(-1) / sn / tn
    payload = dict(version=1, split="train", dataset_fingerprint=fingerprint(records),
        provenance=dict(features_sha256=file_digest(args.features), encoder=args.encoder_description),
        alignment="external_aligned_embeddings_cosine", loss_reduction="token_sum_including_eos",
        pairs=[dict(pair_id=key, clip_score=float(score), caption_nll=float(loss))
               for key, score, loss in zip(ids, scores, losses)])
    QualityTable(payload, records)
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
