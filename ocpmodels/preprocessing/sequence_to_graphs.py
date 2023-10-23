"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional, Union, List

import torch
from torch_geometric.data import Data

from ocpmodels.common.utils import collate


try:
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


class SequenceToGraphs:
    def __init__(
        self,
        r_labels: bool = False,
        # labels could be list or str
        sequence_field: str = None,
        dense_fields: Union[str, List[str]] = None,
        labels: Union[str, List[str]] = None,
    ) -> None:
        self.r_labels = r_labels
        self.labels = [labels] if isinstance(labels, str) else labels
        if self.r_labels:
            assert self.labels is not None
        self.sequence_field = sequence_field
        self.dense_fields = [dense_fields] if isinstance(dense_fields, str) else dense_fields

    def convert(self, atoms: dict, sid=None):
        """Convert a single sequence to a graph.

        Args:
            atoms (ase.atoms.Atoms): A dict of a single sequence.

            sid (uniquely identifying object): An identifier that can be used to track the structure in downstream
            tasks. Common sids used in OCP datasets include unique strings or integers.

        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with natoms,
            and optionally, labels.
        """
        # put the minimum data in torch geometric data object
        if self.sequence_field and (not self.dense_fields):
            data = Data(
                num_nodes=len(atoms[self.sequence_field]),
                sequence=" ".join(list(atoms[self.sequence_field])),
            )
        elif (not self.sequence_field) and self.dense_fields:
            data = Data(
                num_nodes=1,
            )
            for feature in self.dense_fields:
                data[feature] = torch.tensor(atoms[feature], dtype=torch.float32).reshape(1, -1)
        elif self.sequence_field and self.dense_fields:
            data = Data(
                num_nodes=len(atoms[self.sequence_field]),
                sequence=" ".join(list(atoms[self.sequence_field])),
            )
            for feature in self.dense_fields:
                data[feature] = torch.tensor(atoms[feature], dtype=torch.float32).reshape(1, -1)
        else:
            raise NotImplementedError

        # Optionally add a systemid (sid) to the object
        if sid is not None:
            data.sid = sid

        # optionally include other properties
        if self.r_labels:
            for label in self.labels:
                data[label] = torch.tensor(atoms[label])

        return data

    def convert_all(
        self,
        atoms_collection,
        processed_file_path: Optional[str] = None,
        collate_and_save=False,
        disable_tqdm=False,
    ):
        """Convert all atoms objects in a list or in an ase.db to graphs.

        Args:
            atoms_collection (list of dict of sequence):
            processed_file_path (str):
            A string of the path to where the processed file will be written. Default is None.
            collate_and_save (bool): A boolean to collate and save or not. Default is False, so will not write a file.

        Returns:
            data_list (list of torch_geometric.data.Data):
            A list of torch geometric data objects containing molecular graph info and properties.
        """

        # list for all data
        data_list = []
        if isinstance(atoms_collection, list):
            atoms_iter = atoms_collection
        else:
            raise NotImplementedError

        for atoms in tqdm(
            atoms_iter,
            desc="converting sequence atoms collection to graphs",
            total=len(atoms_collection),
            unit=" systems",
            disable=disable_tqdm,
        ):
            data = self.convert(atoms)
            data_list.append(data)

        if collate_and_save:
            data, slices = collate(data_list)
            torch.save((data, slices), processed_file_path)

        return data_list
