import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch

from madewithml.config import mlflow


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    eval("setattr(torch.backends.cudnn, 'deterministic', True)")
    eval("setattr(torch.backends.cudnn, 'benchmark', False)")
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dict(path: str) -> Dict:
    """Load a dictionary from a JSON filepath."""
    with open(path) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, path: str, cls: Any = None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):  # pragma: no cover
        os.makedirs(directory)
    with open(path, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")


def pad_array(arr: np.ndarray, dtype=np.int32) -> np.ndarray:
    """Pad an 2D array with zeros until all rows match the longest row."""
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i][: len(row)] = row
    return padded_arr


def collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Convert a batch of numpy arrays to tensors with appropriate padding."""
    ids = [item["ids"] for item in batch]
    masks = [item["masks"] for item in batch]
    targets = np.array([item["targets"] for item in batch], dtype=np.int64)
    ids = pad_array(np.array(ids, dtype=object))
    masks = pad_array(np.array(masks, dtype=object))
    tensor_batch = {
        "ids": torch.as_tensor(ids, dtype=torch.int64, device=get_device()),
        "masks": torch.as_tensor(masks, dtype=torch.int64, device=get_device()),
        "targets": torch.as_tensor(targets, dtype=torch.int64, device=get_device()),
    }
    return tensor_batch


def get_run_id(experiment_name: str, trial_id: str) -> str:  # pragma: no cover, mlflow functionality
    """Get the MLflow run ID for a specific run name and trial ID."""
    trial_name = f"TorchTrainer_{trial_id}"
    run = mlflow.search_runs(experiment_names=[experiment_name], filter_string=f"tags.trial_name = '{trial_name}'").iloc[0]
    return run.run_id


def dict_to_list(data: Dict, keys: List[str]) -> List[Dict[str, Any]]:
    """Convert a dictionary to a list of dictionaries."""
    list_of_dicts = []
    for i in range(len(data[keys[0]])):
        new_dict = {key: data[key][i] for key in keys}
        list_of_dicts.append(new_dict)
    return list_of_dicts
