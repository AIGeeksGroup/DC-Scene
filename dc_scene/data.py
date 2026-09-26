"""One annotation per sample, with the upstream ScanNet geometry contract."""
import importlib
import random

import numpy as np
import torch


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, scene_dataset, seed=0, deterministic=False):
        self.scenes = scene_dataset
        self.records = scene_dataset.scanrefer
        self.tokenizer = scene_dataset.tokenizer
        self.scene_indices = {name: i for i, name in enumerate(scene_dataset.scan_names)}
        self.seed = seed
        self.deterministic = deterministic
        for record in self.records:
            if record["scene_id"] not in self.scene_indices:
                raise ValueError(f"Annotation scene missing from training split: {record['scene_id']}")
            if not record["token"]:
                raise ValueError("Empty training caption")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        # Scoring uses identical point subsampling for each pair across refreshes.
        np_state, py_state = np.random.get_state(), random.getstate()
        if self.deterministic:
            np.random.seed(self.seed + index)
            random.seed(self.seed + index)
        try:
            sample = self.scenes[self.scene_indices[record["scene_id"]]]
        finally:
            if self.deterministic:
                np.random.set_state(np_state)
                random.setstate(py_state)
        target = (sample["gt_box_object_ids"] == int(record["object_id"])) & sample["gt_box_present"].astype(bool)
        if target.sum() != 1:
            raise ValueError(f"Object {record['object_id']} in {record['scene_id']} has {target.sum()} valid boxes")
        row = int(np.flatnonzero(target)[0])
        # Discard every randomly chosen scene caption: only this annotation trains.
        sample["reference_tokens"][:] = 0
        sample["reference_masks"][:] = 0
        words = record["token"][:self.scenes.max_des_len - 1] + [self.tokenizer.eos_token]
        encoded = self.tokenizer.batch_encode_plus([" ".join(words)],
            max_length=self.scenes.max_des_len, return_tensors="np")
        sample["reference_tokens"][row] = encoded["input_ids"][0]
        sample["reference_masks"][row] = encoded["attention_mask"][0]
        sample["pair_index"] = np.array(index, dtype=np.int64)
        sample["target_box_index"] = np.array(row, dtype=np.int64)
        return sample


def build_datasets(args, scoring=False, evaluation=False):
    module = importlib.import_module(f"datasets.{args.dataset}")
    config = module.DatasetConfig()
    kwargs = dict(num_points=args.num_points, use_color=args.use_color,
                  use_normal=args.use_normal, use_height=args.use_height,
                  use_multiview=args.use_multiview)
    train = module.Dataset(args, config, split_set="train", augment=not scoring, **kwargs)
    pairs = PairDataset(train, seed=args.seed, deterministic=scoring)
    val = module.Dataset(args, config, split_set="val", augment=False, **kwargs) if evaluation else None
    return config, pairs, val


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, args, shuffle=False, epoch=0):
    generator = torch.Generator().manual_seed(args.seed + epoch)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batchsize_per_gpu,
        shuffle=shuffle, num_workers=args.dataset_num_workers, pin_memory=True,
        drop_last=False, worker_init_fn=seed_worker, generator=generator)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
