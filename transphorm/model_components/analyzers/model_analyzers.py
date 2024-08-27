import torch
import lightning as L
from typing import Literal
import matplotlib.pyplot as plt

from typing import Tuple, Protocol
import numpy as np


class ModelAnalyzer(Protocol):
    def compute_random_inference(self): ...
    def plot_data(self): ...
class AutoEncoderAnalyzer:
    """
    A class for analyzing autoencoder models.

    This class provides methods to perform random inference and plot results
    for autoencoder models trained with PyTorch Lightning.

    Attributes:
        model (L.LightningModule): The trained autoencoder model.
        data_module (L.LightningDataModule): The data module used for training and evaluation.
    """

    def __init__(
        self, model: L.LightningModule, data_module: L.LightningDataModule
    ) -> None:
        """
        Initialize the AutoEncoderAnalyzer.

        Args:
            model (L.LightningModule): The trained autoencoder model.
            data_module (L.LightningDataModule): The data module used for training and evaluation.
        """
        self.model = model
        self.data_module = data_module

    def _detatch_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Detach a tensor from the computation graph and convert it to a numpy array.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            np.ndarray: The detached tensor as a numpy array.
        """
        tensor = tensor.to("cpu")
        return tensor.detach().numpy()

    def _get_random_instance(
        self, subset_category: Literal["test", "val", "train"]
    ) -> torch.Tensor:
        """
        Get a random instance from the specified dataset subset.

        Args:
            subset_category (Literal["test", "val", "train"]): The dataset subset to sample from.

        Returns:
            torch.Tensor: A random instance from the specified subset.
        """
        subset = {
            "test": self.data_module.test,
            "val": self.data_module.val,
            "train": self.data_module.train,
        }

        data = subset[subset_category]
        rand_idx = np.random.randint(0, len(data))

        return data[rand_idx][0].unsqueeze(0)

    def compute_random_inference(
        self, subset_category: Literal["test", "val", "train"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute inference on a random instance from the specified subset.

        Args:
            subset_category (Literal["test", "val", "train"]): The dataset subset to sample from.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the input and reconstructed output as numpy arrays.
        """
        x_test = self._get_random_instance(subset_category)
        with torch.no_grad():
            x_hat_test = self.model(x_test)
        x_test_samp = self._detatch_tensor(x_test.squeeze(0))
        x_hat_test_samp = self._detatch_tensor(x_hat_test.squeeze(0))
        return x_test_samp, x_hat_test_samp

    def plot_data(self, x: np.ndarray, x_hat: np.ndarray) -> None:
        """
        Plot the original and reconstructed data.

        Args:
            x (np.ndarray): The original input data.
            x_hat (np.ndarray): The reconstructed output data.
        """
        fig, ax = plt.subplot_mosaic(mosaic="ABC", figsize=(18, 6))
        ax["A"].plot(x[0], color="k")
        ax["A"].plot(x_hat[0], color="r")
        ax["B"].plot(x[1], color="k")
        ax["B"].plot(x_hat[1], color="r")
        ax["C"].plot(x[2], color="k")
        ax["C"].plot(x_hat[2], color="r")
        ax["A"].set_title("Signal")
        ax["B"].set_title("Timestamps vector 1")
        ax["C"].set_title("Timestamps vector 2")


class AutoEncoderAnalyzer:
    def __init__(self, model: L.LightningModule, data_module: L.LightningDataModule):
        self.model = model
        self.data_module = data_module

    def _detatch_tensor(self, tensor: torch.Tensor):
        tensor.to("cpu")
        return tensor.detach().numpy()

    def _get_random_instance(self, subset_category: Literal["test", "val", "train"]):
        subset = {
            "test": self.data_module.test,
            "val": self.data_module.val,
            "train": self.data_module.train,
        }

        data = subset[subset_category]
        rand_idx = np.random.randint(0, len(data))

        return data[rand_idx][0].unsqueeze(0)

    def compute_random_inference(self, subset_category):
        x_test = self._get_random_instance(subset_category)
        with torch.no_grad():
            x_hat_test = self.model(x_test)
        x_test_samp = self._detatch_tensor(x_test.squeeze(0))
        x_hat_test_samp = self._detatch_tensor(x_hat_test.squeeze(0))
        return x_test_samp, x_hat_test_samp

    def plot_data(self, x, x_hat):
        fig, ax = plt.subplot_mosaic(mosaic="ABC", **{"figsize": (18, 6)})
        ax["A"].plot(x[0], color="k")
        ax["A"].plot(x_hat[0], color="r")
        ax["B"].plot(x[1], color="k")
        ax["B"].plot(x_hat[1], color="r")
        ax["C"].plot(x[2], color="k")
        ax["C"].plot(x_hat[2], color="r")
        ax["A"].title.set_text("Signal")
        ax["B"].title.set_text("Timestamps vector 1")
        ax["C"].title.set_text("Timestamps vector 2")
