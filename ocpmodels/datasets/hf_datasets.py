"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import logging
import pickle
import warnings
from pathlib import Path
from typing import List, Optional, TypeVar

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from datasets import Dataset
from transformers import AutoTokenizer

from ocpmodels.common.registry import registry
from ocpmodels.common.typing import assert_is_instance
from ocpmodels.common.utils import pyg2_data_transform
from ocpmodels.datasets.target_metadata_guesser import guess_property_metadata


@registry.register_dataset("huggingface_datasets")
class HuggingfaceDataset(Dataset):
    def __new__(cls, config, *args, **kwargs):
        path = Path(config["src"])
        if not path.is_file():
            db_paths = [str(path) for path in sorted(path.glob("*.csv"))]
            assert len(db_paths) > 0, f"No csv found in '{path}'"
            # self.metadata_path = self.path / "metadata.npz"
        else:
            db_paths = str(path)
            # self.metadata_path = self.path.parent / "metadata.npz"
        return Dataset.from_csv(db_paths, **{i: config[i] for i in config if i != "src"})

