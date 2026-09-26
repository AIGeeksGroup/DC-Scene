"""Create scene lists for official ScanRefer JSON or convert Nr3D CSV."""
import argparse
import csv
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["scanrefer", "nr3d"], required=True)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--metadata_root", type=Path, default=Path("data/scannet/meta_data"))
    args = parser.parse_args()
    splits = {split: set((args.metadata_root / f"scannetv2_{split}.txt").read_text().split()) for split in ["train", "val"]}
    if splits["train"] & splits["val"]:
        raise ValueError("ScanNet scene splits overlap")
    records = {"train": [], "val": []}
    if args.dataset == "nr3d":
        with (args.data_root / "nr3d.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                scene = row["scan_id"]
                entry = dict(scene_id=scene, object_id=str(row["target_id"]),
                             object_name=row["instance_type"], ann_id=str(row["assignmentid"]),
                             description=row["utterance"].lower(),
                             token=re.findall(r"[a-z0-9]+|[^\w\s]|_", row["utterance"].lower()))
                for split in records:
                    if scene in splits[split]:
                        records[split].append(entry)
    else:
        for split in records:
            records[split] = json.loads((args.data_root / f"ScanRefer_filtered_{split}.json").read_text())
    prefix = "ScanRefer_filtered" if args.dataset == "scanrefer" else "nr3d"
    for split, entries in records.items():
        if not entries:
            raise ValueError(f"Empty {split} annotations")
        names = sorted({entry["scene_id"] for entry in entries})
        if set(names) - splits[split]:
            raise ValueError(f"{split} annotations contain scenes outside official split")
        for entry in entries:
            for key in ("scene_id", "object_id", "object_name", "ann_id", "token"):
                if key not in entry:
                    raise ValueError(f"Annotation missing {key}")
        if args.dataset == "nr3d":
            (args.data_root / f"{prefix}_{split}.json").write_text(json.dumps(entries, indent=2) + "\n")
        (args.data_root / f"{prefix}_{split}.txt").write_text("\n".join(names) + "\n")
        print(f"{split}: {len(entries)} pairs in {len(names)} scenes")


if __name__ == "__main__":
    main()
