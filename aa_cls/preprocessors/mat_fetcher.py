import polars as pl
import scipy.io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal
import pyarrow as pa
import pyarrow.parquet as pq


from typing import List, Union, Dict, Any

import torch


class MatFetcher:
    """
    A class for fetching, processing, and storing .mat files containing trace data.

    Attributes:
        path (Path): The directory path to search for .mat files.
        mat_files (List[Path]): List of paths to .mat files.
        data (np.ndarray): Processed data from .mat files.
    """

    def __init__(self, path: Union[str, Path], day_filter: int = 0):
        """
        Initialize the MatFetcher object.

        Args:
            path (Union[str, Path]): The directory path to search for .mat files.
        """
        self.path = Path(path)
        self.mat_files: List[Path] = None
        self.data: np.ndarray = None
        self.day_filter = day_filter

    def _fetch_mat_files(self) -> None:
        """
        Fetch all .mat files in the specified directory and its subdirectories.
        """
        files = [f for f in self.path.rglob("*.mat")]
        self.mat_files = files

    def _get_day_int(self, path: Path) -> int:
        name = path.parent.name
        day_int = int(name.strip("[dD]ay"))
        return day_int

    def _filter_by_day(self):
        files = [f for f in self.mat_files if self._get_day_int(f) >= self.day_filter]
        self.mat_files = files

    def _label_behavior(self, label: str) -> Literal[0, 1]:
        """
        Label the behavior based on the file name.

        Args:
            label (str): The label string from the file name.

        Returns:
            Literal[0, 1]: 1 if 'Escape' is in the label, 0 otherwise.
        """
        if "Escape" in label:
            return 1.0
        else:
            return 0.0

    def _read_mat_file(self, path: Path) -> np.ndarray:
        """
        Read a .mat file and extract the trace data and label.

        Args:
            path (Path): Path to the .mat file.

        Returns:
            np.ndarray: Array containing labels and traces.
        """
        data = scipy.io.loadmat(path)
        for k, v in data.items():
            if "data" in k:
                traces = v
                label = self._label_behavior(k)
        labels = np.array([label for _ in range(traces.shape[0])])
        return np.column_stack([labels, traces])

    def load_data(self) -> None:
        """
        Load data from all .mat files in the specified directory.
        """
        self._fetch_mat_files()
        self._filter_by_day()

        data_arrs = [self._read_mat_file(f) for f in self.mat_files]
        self.data = np.concatenate(data_arrs, axis=0)

    def write_to_torch(self):
        labels = self.data[:, 0]
        traces = self.data[:, 1:]

        labels_path = self.path / f"labels_start_day_{self.day_filter}.pt"
        traces_path = self.path / f"traces_start_day_{self.day_filter}.pt"

        torch.save(torch.tensor(labels), labels_path)
        torch.save(torch.tensor(traces), traces_path)
