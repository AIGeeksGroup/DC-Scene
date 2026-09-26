"""Train-only, pair-level DIQ selection (paper equations 2--5).

The manuscript does not specify a scalar DIQ ranking. We use the mean of
percentile ranks of alignment and summed caption NLL inside the DIQ rectangle.
This explicit policy favors reliable, informative pairs after outlier removal.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def fingerprint(records):
    payload = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def pair_id(record):
    # Include content: the same annotation ID with edited text is not the same pair.
    return fingerprint({k: record[k] for k in ("scene_id", "object_id", "ann_id", "token")})


def percent_ranks(values):
    """Average ranks for ties, independent of annotation input ordering."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    ranks = starts + (counts - 1) / 2
    return ranks[inverse] / max(len(values) - 1, 1)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


class QualityTable:
    def __init__(self, payload, records):
        if not records:
            raise ValueError("Empty training annotation file")
        if payload.get("version") != 1 or payload.get("split") != "train":
            raise ValueError("Quality scores must have version=1 and split=train")
        if payload.get("dataset_fingerprint") != fingerprint(records):
            raise ValueError("Scores do not match the full training annotation file")
        rows = payload["pairs"]
        lookup = {row["pair_id"]: row for row in rows}
        ids = [pair_id(record) for record in records]
        if len(lookup) != len(rows) or len(set(ids)) != len(ids) or set(ids) != set(lookup):
            raise ValueError("Quality scores require exactly one row per unique training pair")
        self.ids = ids
        self.records = records
        self.scores = np.array([lookup[key]["clip_score"] for key in ids], dtype=np.float64)
        self.losses = np.array([lookup[key]["caption_nll"] for key in ids], dtype=np.float64)
        if not np.isfinite(self.scores).all() or not np.isfinite(self.losses).all() or (self.losses < 0).any():
            raise ValueError("Nonfinite scores or negative caption NLL")
        self.digest = fingerprint(payload)

    @classmethod
    def load(cls, path, records):
        return cls(json.loads(Path(path).read_text()), records)

    def rank(self, strategy="diq", lower=0.05, upper=0.95, alignment_weight=0.5,
             seed=0, top_k_per_scene=None):
        if not 0 <= lower < upper <= 1 or not 0 <= alignment_weight <= 1:
            raise ValueError("Invalid quantile bounds or alignment weight")
        smin, smax = np.quantile(self.scores, [lower, upper])
        lmin, lmax = np.quantile(self.losses, [lower, upper])
        eligible = np.ones(len(self.ids), dtype=bool)
        if strategy == "diq":
            eligible &= (self.scores >= smin) & (self.scores <= smax)
            eligible &= (self.losses >= lmin) & (self.losses <= lmax)
        elif strategy not in ("clip", "loss", "random", "full"):
            raise ValueError(f"Unknown selection strategy: {strategy}")
        indices = np.flatnonzero(eligible)
        if not len(indices):
            raise ValueError("No pairs survive the DIQ rectangle")
        s = percent_ranks(self.scores[indices])
        loss = percent_ranks(self.losses[indices])
        if strategy == "diq":
            priority = alignment_weight * s + (1 - alignment_weight) * loss
        elif strategy == "clip":
            priority = s
        elif strategy == "loss":
            priority = loss
        else:
            priority = np.random.default_rng(seed).random(len(indices))
        ordered = sorted(zip(indices.tolist(), priority.tolist()), key=lambda x: (-x[1], self.ids[x[0]]))
        if top_k_per_scene is not None:
            if top_k_per_scene < 1:
                raise ValueError("top_k_per_scene must be positive")
            counts, capped = {}, []
            for index, score in ordered:
                scene = self.records[index]["scene_id"]
                if counts.get(scene, 0) < top_k_per_scene:
                    capped.append((index, score))
                    counts[scene] = counts.get(scene, 0) + 1
            ordered = capped
        self.bounds = dict(score_min=float(smin), score_max=float(smax), loss_min=float(lmin), loss_max=float(lmax))
        return [index for index, _ in ordered]


class Curriculum:
    def __init__(self, ranked_indices, population, ratios=(0.25, 0.5, 0.75),
                 stage_epochs=(360, 360, 360)):
        if not ratios or len(ratios) != len(stage_epochs):
            raise ValueError("Each stage needs a ratio and duration")
        if any(not 0 < r <= 1 for r in ratios) or list(ratios) != sorted(ratios):
            raise ValueError("Ratios must be nondecreasing and in (0, 1]")
        if any(not isinstance(e, int) or e <= 0 for e in stage_epochs):
            raise ValueError("Stage durations must be positive integers")
        self.ranked = list(ranked_indices)
        self.population = population
        self.ratios = list(ratios)
        self.stage_epochs = list(stage_epochs)
        self.ends = np.cumsum(stage_epochs).tolist()
        self.sizes = [max(1, math.ceil(population * r)) for r in ratios]
        if max(self.sizes) > len(self.ranked):
            raise ValueError(f"Requested {max(self.sizes)} pairs, but only {len(self.ranked)} qualify. "
                             "Widen quantiles, relax the per-scene cap, or lower the ratios explicitly.")

    def stage(self, epoch):
        if not 0 <= epoch < self.ends[-1]:
            raise ValueError("Epoch is outside curriculum")
        return int(np.searchsorted(self.ends, epoch, side="right"))

    def indices(self, epoch):
        return self.ranked[:self.sizes[self.stage(epoch)]]

    def manifest(self, ids):
        return {"population": self.population, "ratios": self.ratios,
                "stage_epochs": self.stage_epochs, "sizes": self.sizes,
                "relative_sample_budget": sum(n * e for n, e in zip(self.sizes, self.stage_epochs)) / (self.population * self.ends[-1]),
                "ranked_pair_ids": [ids[i] for i in self.ranked]}
