from abc import ABC
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import h5py
import numpy as np
from torch.utils.data import Dataset


class HDF5Dataset(Dataset, ABC):
    """
    A dataset class representing data stored in one or more HDF5 files.

    Parameters:
        paths (Iterable[str]): Iterable containing paths to HDF5 files.
        data_key (str): Key representing the dataset in the HDF5 files.
        preload_data (bool): If True, preload all data into memory. Defaults to False.

    Attributes:
        paths (List[Path]): List of paths to HDF5 files.
        preload (bool): Indicates whether data is preloaded into memory.
        data (Dict[str, np.ndarray]): Dictionary containing preloaded data.
        lengths (List[int]): List of lengths of datasets in each HDF5 file.
        cum_idx (numpy.ndarray): Cumulative indices used for indexing datasets across multiple files.

    Methods:
        __len__(): Returns the total length of the concatenated datasets.
        __del__(): Closes all opened HDF5 files if not preloaded.
        _read_recursive_file(file, file_idx, parent_name): Recursively reads data from HDF5 groups and datasets.
        _read_recursive_data(data, idx): Recursively reads preloaded data.
        __getitem__(idx): Retrieves data at a specified index.
        get_all(name): Retrieves all values for a given dataset name.
    """

    def __init__(self, paths: Iterable[str], data_key: str, preload_data: bool = False):
        """
        Initializes `HDF5Dataset` with paths to HDF5 files and the data key.

        Args:
            paths (Iterable[str]): Iterable containing paths to HDF5 files.
            data_key (str): Key representing the dataset in the HDF5 files.
            preload_data (bool): If True, preload all data into memory. Defaults to False.
        """
        super().__init__()
        self.paths = [Path(p) for p in paths]
        self.preload = preload_data
        self.lengths = []
        if self.preload:
            # self.manager = mp.Manager()
            self.shared_data = {}  # self.manager.dict()
        for h5path in self.paths:
            with h5py.File(h5path, "r") as hf:
                self.lengths.append(len(hf[data_key]))
                if self.preload:
                    self.shared_data[h5path.stem] = self._read_recursive_file(hf)

        self.cum_idx = np.hstack((0, np.cumsum(self.lengths)[:-1]))

    def __len__(self) -> int:
        """
        Returns the total length of the concatenated datasets.

        Returns:
            int: Total length of the concatenated datasets.
        """
        return int(np.sum(self.lengths))

    def _read_recursive_file(self, file: h5py.File, file_idx: Optional[int] = None, parent_name: str = "") -> Dict[str, Any]:
        """
        Recursively reads data from HDF5 groups and datasets.

        Args:
            file (h5py.File): HDF5 file to read data from.
            file_idx (Optional[int]): Index to read data from.
            parent_name (str, optional): Parent name for recursive reading. Defaults to "".

        Returns:
            Dict[str, Any]: Dictionary containing the read data.
        """
        result = {}
        for key in file.keys():
            if isinstance(file[key], h5py.Group):
                result[key] = self._read_recursive_file(file[key], file_idx, parent_name=f"{parent_name}/{key}")
            elif isinstance(file[key], h5py.Dataset):
                if file_idx is not None:
                    result[key] = file[f"{parent_name}/{key}"][file_idx]
                else:
                    result[key] = file[f"{parent_name}/{key}"][:]
        return result

    def _read_recursive_data(self, data: Dict, idx: int) -> Dict[str, Any]:
        """
        Recursively reads preloaded data.

        Args:
            data (Dict): Preloaded data.
            idx (int): Index to read data from.

        Returns:
            Dict[str, Any]: Dictionary containing the read data.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = self._read_recursive_data(data[key], idx)
            else:
                result[key] = data[key][idx]
        return result

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves data at a specified index.

        Args:
            idx (int): Index to retrieve data from.

        Returns:
            Dict[str, Any]: Dictionary containing the retrieved data.
        """
        file_number = np.searchsorted(self.cum_idx, idx, side="right") - 1
        path = self.paths[file_number]
        file_idx = idx - self.cum_idx[file_number]

        if self.preload:
            data = self._read_recursive_data(self.shared_data[path.stem], file_idx)
        else:
            with h5py.File(path, "r") as file:
                data = self._read_recursive_file(file, file_idx)

        data.update({"dataset": path.stem, "file_idx": file_idx, "total_idx": idx})

        return data

    def get_all(self, name: str) -> np.ndarray:
        """
        Retrieves all values for a given dataset name.

        Args:
            name (str): Name of the dataset.

        Returns:
            numpy.ndarray: Array containing all values of the dataset or group.
        """
        target = []

        if name == "dataset":
            for n, length in enumerate(self.lengths):
                target.extend([n] * length)
        else:
            if self.preload:
                for key, data in self.shared_data.items():
                    target.extend(self.shared_data[key][name])
            else:
                for path in self.paths:
                    with h5py.File(path, "r") as file:
                        target.extend(file[name][:])

        return np.asarray(target)
