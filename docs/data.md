# Data and feature preparation

## ScanNet geometry

Obtain ScanNet v2 through its official access procedure; the repository does not distribute scans. Use the upstream preprocessing included under `data/scannet`. From the repository root:

```bash
(cd data/scannet && SCANNET_RAW_ROOT=/path/to/scans \
  SCANNET_OUTPUT_ROOT=/path/to/scannet_data python batch_load_scannet_data.py)
```

For each scene the loader needs:

| File | Shape / contents |
|---|---|
| `{scene}_aligned_vert.npy` | `[N,9]`: aligned XYZ, RGB (0–255), normal XYZ |
| `{scene}_ins_label.npy` | `[N]`: instance labels |
| `{scene}_sem_label.npy` | `[N]`: NYU40 semantic labels |
| `{scene}_aligned_bbox.npy` | `[M,8]`: center XYZ, extent XYZ, semantic ID, object ID |

The supported ScanNet contract has at most 128 retained object boxes. Use upstream object-ID conventions consistently with ScanRefer/Nr3D annotations. Defaults sample 40,000 points and use RGB, normals, and height. Optional multiview features come from `enet_feats_maxpool.hdf5` under `scan_root`, keyed by scene with shape `[N,128]`; features must be aligned to the same vertex rows. Prediction takes these features as a separate `.npy` array.

## Language annotations

Download the official [ScanRefer annotations](https://github.com/daveredrum/ScanRefer) and/or [Nr3D annotations](https://referit3d.github.io/). Keep the official scene split; do not compute quality percentiles from validation captions.

Each JSON annotation contains:

```json
{
  "scene_id": "scene0000_00",
  "object_id": "0",
  "object_name": "chair",
  "ann_id": "0",
  "description": "a chair near the table",
  "token": ["a", "chair", "near", "the", "table"]
}
```

This is a schema example, not a real released annotation. IDs and tokens jointly define a stable hashed `pair_id`. An edited caption is a different pair. Duplicate identities are rejected when loading quality tables.

For ScanRefer, place `ScanRefer_filtered_train.json` and `ScanRefer_filtered_val.json` under `data_root`; `scripts/prepare_annotations.py --dataset scanrefer` creates the corresponding `.txt` scene lists. For Nr3D, place `nr3d.csv` under `data_root`; `--dataset nr3d` converts it to train/val JSON and scene lists using the bundled official ScanNet split, following Vote2Cap's lowercase tokenization. Use the supplied upstream vocabularies with upstream checkpoints. The converter does not silently rebuild them.

The loaders read both official train and validation metadata for training/scoring/evaluation. Raw geometry is loaded on demand. Prediction only needs the vocabulary, ScanNet class metadata, and point cloud. New annotations must include object IDs that exist in the processed boxes.

The pair dataset loads a scene but zeroes all caption masks except the selected annotation's target object. Thus 25% means 25% of scene–caption pairs, even if some scenes contain more captions than others. Detection supervision still uses the complete scene boxes.

## Quality table

`main.py --mode score` produces:

```json
{
  "version": 1,
  "split": "train",
  "dataset_fingerprint": "SHA256 of the complete training JSON records",
  "provenance": {"checkpoint_sha256": "SHA256 of the scoring checkpoint"},
  "alignment": "pretrained_coca_joint_latents_cosine",
  "loss_reduction": "token_sum_including_eos",
  "pairs": [
    {"pair_id": "stable pair hash", "clip_score": 0.42, "caption_nll": 17.8}
  ]
}
```

These numbers illustrate the schema only. The score table must cover every training annotation exactly once; this example is not a usable quality file. The loader rejects duplicates, missing/extra pairs, changed annotations, nonfinite indicators, negative NLL, and non-training tables. Preserve scorer checkpoint and preprocessing provenance when moving tables between experiments.

## External quality features

If the original trained CLIP-aligned scene/text embeddings and decoder losses are available, package them in an `.npz` archive without pickled objects:

- `pair_ids`: Unicode strings `[N]`, computed with `dc_scene.quality.pair_id(record)`.
- `scene_embeddings`: `[N,D]`, pretrained scene representations.
- `text_embeddings`: `[N,D]`, corresponding caption representations in the **same trained latent space**.
- `caption_nll`: `[N]`, teacher-forced **sum** over non-padding target tokens, including EOS.

The row order may differ from the annotations; matching is by pair ID. Norms must be nonzero. Raw CLIP text embeddings and arbitrary point-cloud features are not interchangeable, even if both have dimension D. Do not fit a new random projector merely to satisfy the shape contract.

```bash
python scripts/import_quality.py \
  --annotations data/ScanRefer_filtered_train.json \
  --features /path/to/aligned_train_features.npz \
  --encoder_description "checkpoint identifier, encoder and preprocessing settings" \
  --output outputs/quality/3dcoca_scanrefer.json
```

Only supply training annotations/features here; the import cannot infer whether externally supplied records were mislabeled. The training loader subsequently verifies the table against its official training metadata.
