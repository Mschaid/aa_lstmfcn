from typing import Callable, Dict
import torch
import torch.nn as nn

import numpy as np


class FeatureExtractor:
    """
    FeatureExtractor is a utility class to extract intermediate features from a given model.

    This class registers hooks to specified layers of the model to capture their output during
    the forward pass. The extracted features can be accessed via the `extract` method.

    Attributes:
        model (nn.Module): The model from which features are to be extracted.
        features (dict): A dictionary to store the extracted features.
    """

    def __init__(self, model):
        self.model = model
        self.features = {}

    def extract_features(self, name) -> Callable:
        """
        Registers a hook to a specified layer of the model to capture its output.

        This method creates a hook function that will be called during the forward pass of the model.
        The hook function captures the output of the specified layer and stores it in the `features`
        dictionary with the given name as the key.

        Args:
            name (str): The name to be used as the key in the `features` dictionary for storing the output.

        Returns:
            function: A hook function that can be registered to a layer of the model.
        """

        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            self.features[name] = output.detach().numpy()

        return hook

    def extract(self, x) -> Dict[str, np.array]:
        """
        Extract features from the input tensor using the model.

        This method clears the previously stored features and then performs a
        forward pass through the model without computing gradients.

        Args:
            x (torch.Tensor): The input tensor to extract features from.

        Returns:
            dict: A dictionary containing the extracted features as np.array - for plotting purposes as torch tensors are not compatible.
        """
        # clears features from past layer
        self.features.clear()
        with torch.no_grad():

            self.model(x)
        return self.features
