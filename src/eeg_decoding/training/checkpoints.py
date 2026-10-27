from pathlib import Path
import random
import numpy as np
import torch


def save_checkpoint(path, model, optimizer=None, scaler=None, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "metadata": metadata,
             "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
             "numpy_rng": {"kind": np.random.get_state()[0], "keys": np.random.get_state()[1].tolist(),
                           "position": np.random.get_state()[2], "has_gauss": np.random.get_state()[3],
                           "cached_gaussian": np.random.get_state()[4]}}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)

def load_checkpoint(path, model=None, optimizer=None, scaler=None, restore_rng=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if model is not None:
        model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        n = state["numpy_rng"]
        np.random.set_state((n["kind"], np.asarray(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"]))
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["metadata"]
