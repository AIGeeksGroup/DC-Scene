"""Atomic epoch-boundary checkpoints, including the curriculum and RNG states."""
import hashlib
import random
from pathlib import Path

import numpy as np
import torch


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_weights(model, path, allow_partial=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not allow_partial and checkpoint.get("signature", model.signature) != model.signature:
        raise ValueError("Checkpoint vocabulary, input features, or backbone settings differ from the model")
    weights = checkpoint.get("model", checkpoint)
    if allow_partial:
        current = model.state_dict()
        compatible = {key: value for key, value in weights.items()
                      if key in current and current[key].shape == value.shape}
        if not compatible:
            raise ValueError("No checkpoint parameters match this model")
        result = model.load_state_dict(compatible, strict=False)
        print(f"Warm start: loaded {len(compatible)} tensors, missing {len(result.missing_keys)}, "
              f"skipped {len(weights) - len(compatible)}. This is not an exact resume.")
    else:
        model.load_state_dict(weights, strict=True)
    return checkpoint


def save(path, model, optimizer, epoch, step, samples, best, manifest, args):
    np_state = np.random.get_state()
    payload = dict(model=model.state_dict(), signature=model.signature, optimizer=optimizer.state_dict(),
        epoch=epoch, step=step, samples=samples, best=best, manifest=manifest, args=vars(args),
        rng=dict(python=random.getstate(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all(),
                 numpy=[np_state[0], np_state[1].tolist(), *np_state[2:]]))
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_rng(checkpoint):
    state = checkpoint["rng"]
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:]))
