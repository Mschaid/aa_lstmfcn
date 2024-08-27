from typing import Tuple
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data.dataset import random_split
import torch
import lightning as L
from torch.utils.data import Dataset, TensorDataset, DataLoader
from pathlib import Path
import numpy as np
from scipy import signal


class AADataModule(L.LightningDataModule):
    def __init__(self, main_path: Path, batch_size: int, num_workers: int = 1):
        super().__init__()
        self.main_path = main_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = None

    def store_input_size(self, value):
        self.input_size = value

    def prepare_data(self):
        data = torch.load(self.main_path)

        data = data[~torch.isnan(data[:, 0])]

        features = data[:, 1:]
        labels = data[:, 0]

        input_size = features[0].shape[0]
        self.store_input_size(input_size)

        features = features.view(-1, 1, self.input_size)

        labels = labels.view(-1, 1)
        self.data = TensorDataset(
            features,
            labels,
        )

    def setup(self, stage):
        torch.manual_seed(42)
        self.train, self.val, self.test = random_split(self.data, [0.7, 0.15, 0.15])

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    @property
    def tensor_shape(self):
        seq = self.train.dataset.tensors
        return seq[0].shape


class AATrialDataModule(L.LightningDataModule):
    def __init__(self, main_path: Path, batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.main_path = main_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = None

    def _filter_time_window(self, traces, time_window: Tuple[int, int]):
        time_range = torch.arange(-25, 20, 45 / 45777)
        min_time = time_window[0]
        max_time = time_window[1]
        idx_min = torch.argmin(torch.abs(time_range - min_time)).item()
        idx_max = torch.argmin(torch.abs(time_range - max_time)).item()
        filtered_traces = traces[:, idx_min:idx_max]
        return filtered_traces

    def _get_non_nan_idx(self, traces):
        idx = torch.nonzero(~traces.isnan().any(dim=1)).squeeze()
        return idx

    def _low_pass_filter(self, data: torch.Tensor, cutoff=1, order=1):
        np_arr = data.detach().numpy()

        nyq = 0.5 * 1017  # nyquist is half sampling rate
        norm_cutoff = cutoff / nyq
        b, a = signal.butter(order, norm_cutoff, btype="low", analog=False)
        filtered_data = signal.filtfilt(b, a, np_arr)
        filtered_data_copy = filtered_data.copy()
        return torch.Tensor(filtered_data_copy)

    def _find_path(self, key_word):
        for f in self.main_path.iterdir():
            if key_word in f.name:
                return f

    def prepare_data(
        self, time_window: Tuple[int, int] = (-1, 5), low_pass_filter=True
    ):

        traces = torch.load(self._find_path("trace")).to(torch.float32)
        labels = torch.load(self._find_path("label")).to(torch.float32)

        non_nan_idx = self._get_non_nan_idx(traces)
        non_na_traces = traces[non_nan_idx]
        non_nan_labels = labels[non_nan_idx]

        time_filtered_traces = self._filter_time_window(
            non_na_traces, time_window=time_window
        )

        # filtered_traces = filtered_traces[:, ::1]
        filtered_traces_3d = time_filtered_traces.view(
            -1, 1, time_filtered_traces.size(1)
        )

        if low_pass_filter:
            filtered_traces_3d = self._low_pass_filter(filtered_traces_3d)

        self.data = TensorDataset(
            filtered_traces_3d,
            non_nan_labels,
        )

    def setup(self, stage):
        torch.manual_seed(42)
        self.train, self.val, self.test = random_split(self.data, [0.7, 0.15, 0.15])

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    @property
    def tensor_shape(self):
        seq = self.train.dataset.tensors
        return seq[0].shape
