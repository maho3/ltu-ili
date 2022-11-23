"""Script to generate train/test/val split idx
"""
from typing import Dict, List
import numpy as np
import random
import json


def get_split(
    n_nodes: int, percent_test: float, percent_val: float
) -> Dict[str, List[int]]:
    """Get idx for train test split given a number of nodes

    Args:
        n_nodes (int): number of total simulations
        percent_test (float): percentage in test set
        percent_val (float): percentage in validation set

    Returns:
        Dict[str, List[int]]: dictionary with idx
    """
    idx = list(range(n_nodes))
    random.shuffle(idx)
    n_val = int(np.floor(percent_val * n_nodes))
    n_test = int(np.floor(percent_test * n_nodes))
    val_idx = idx[:n_val]
    test_idx = idx[n_val : n_val + n_test]
    train_idx = list(idx[n_val + n_test :])
    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


if __name__ == "__main__":
    random.seed(10)
    filename = "quijote_train_test_val.json"
    n_nodes = 2000
    percent_test = 0.1
    percent_val = 0.1
    idx_dict = get_split(
        n_nodes=n_nodes, percent_test=percent_test, percent_val=percent_val
    )
    with open(filename, "w") as f:
        json.dump(idx_dict, f)
