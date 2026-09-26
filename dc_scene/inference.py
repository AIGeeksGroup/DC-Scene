"""Dense captions for a preprocessed ScanNet-style point cloud, without labels."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .quality import atomic_json


def inference_metadata(args):
    module = importlib.import_module(f"datasets.{args.dataset}")
    name = "ScanRefer_vocabulary.json" if args.dataset == "scene_scanrefer" else "nr3d_vocabulary.json"
    vocabulary = json.loads((Path(args.data_root) / name).read_text())
    tokenizer = module.ScanReferTokenizer(vocabulary["word2idx"])
    tokenizer.pad_token = tokenizer.eos_token
    return module.DatasetConfig(), SimpleNamespace(tokenizer=tokenizer, scenes=SimpleNamespace(max_des_len=args.max_des_len))


@torch.no_grad()
def predict(args, model):
    from utils.proposal_parser import parse_predictions
    vertices = np.load(args.point_cloud, allow_pickle=False)
    required = 9 if args.use_normal else (6 if args.use_color else 3)
    if vertices.ndim != 2 or vertices.shape[1] < required or len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError(f"Expected finite point cloud [N, >={required}] with xyz,rgb,normal columns")
    features = [vertices[:, :3].astype(np.float32)]
    if args.use_color:
        features.append((vertices[:, 3:6] - np.array([109.8, 97.2, 83.8])) / 256.)
    if args.use_normal:
        features.append(vertices[:, 6:9])
    if args.use_multiview:
        if not args.multiview_features:
            raise ValueError("--use_multiview requires --multiview_features")
        multiview = np.load(args.multiview_features, allow_pickle=False)
        if multiview.shape != (len(vertices), 128) or not np.isfinite(multiview).all():
            raise ValueError("Multiview features must have finite shape [N,128]")
        features.append(multiview)
    if args.use_height:
        features.append(vertices[:, 2:3] - np.percentile(vertices[:, 2], .99))
    points = np.concatenate(features, axis=1).astype(np.float32)
    choices = np.random.default_rng(args.seed).choice(len(points), args.num_points, replace=len(points) < args.num_points)
    points = torch.from_numpy(points[choices]).unsqueeze(0).to(next(model.parameters()).device)
    batch = dict(point_clouds=points, point_cloud_dims_min=points[..., :3].amin(1),
                 point_cloud_dims_max=points[..., :3].amax(1))
    model.eval()
    outputs = model(batch, is_eval=True)
    keep = parse_predictions(outputs["box_corners"], outputs["sem_cls_prob"],
                             outputs["objectness_prob"], points)[0]
    objects = []
    for i in np.flatnonzero(keep):
        if outputs["sem_cls_logits"][0, i].argmax().item() == outputs["sem_cls_logits"].shape[-1] - 1:
            continue
        objects.append(dict(proposal_id=int(i), caption=outputs["lang_cap"][0][i],
            center=outputs["center_unnormalized"][0, i].tolist(),
            size=outputs["size_unnormalized"][0, i].tolist(),
            box_corners=outputs["box_corners"][0, i].tolist(),
            semantic_class=int(outputs["sem_cls_prob"][0, i].argmax()),
            objectness=float(outputs["objectness_prob"][0, i])))
    atomic_json(args.output, dict(point_cloud=args.point_cloud,
        center_coordinate_system="ScanNet upright xyz", box_corners_coordinate_system="camera xyz (x,-z,y)", objects=objects))
