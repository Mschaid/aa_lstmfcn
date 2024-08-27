import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import structlog
from dotenv import load_dotenv
from joblib import load
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier

from aa_cls.experiments.aa_classifiers.aa_lstmfcn_bayes_5_day import load_data


def main() -> None:
    """
    Main function to run inference on the trained LSTM-FCN model.

    This function loads the trained model, processes the data, runs inference,
    and saves the results.
    """
    log: structlog.BoundLogger = structlog.get_logger()
    load_dotenv()
    MODEL_PATH: Path = Path(
        "/projects/p31961/aa_cls/models/aa_classifiers/sk_models/accepted_plywood_8946.joblib"
    )
    DATA_PATH: Path = Path(os.getenv("DATA_PATH_5_DAY", ""))
    path_to_save: Path = Path("/home/mds8301/data/gaby_data/over_day_5/eval_data")

    log.info("Loading model")
    model: LSTMFCNClassifier = load(MODEL_PATH)

    log.info("Loading and processing data")
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    model.fit(X_train, y_train)

    log.info("Running inference")
    results: Dict[str, np.ndarray] = {
        "x_train": X_train,
        "x_test": X_test,
        "y_train": y_train,
        "y_train_pred": model.predict(X_train),
        "y_test": y_test,
        "y_test_pred": model.predict(X_test),
    }

    log.info("Saving data")
    save_results(results, path_to_save)

    log.info("Saving model")
    save_model(
        model,
        "/projects/p31961/aa_cls/models/aa_classifiers/sk_models/accepted_plywood_8946",
    )

    log.info("Inference complete")


def save_results(results: Dict[str, np.ndarray], path_to_save: Path) -> None:
    """
    Save the inference results to individual .npy files and a combined .npz file.

    Args:
        results (Dict[str, np.ndarray]): Dictionary containing the inference results.
        path_to_save (Path): Path to the directory where results will be saved.
    """
    for k, v in results.items():
        np.save(path_to_save / f"{k}.npy", v)
    np.savez(path_to_save / "inference_results.npz", **results)


def save_model(model: LSTMFCNClassifier, path: str) -> None:
    """
    Save the trained model to the specified path.

    Args:
        model (LSTMFCNClassifier): The trained LSTM-FCN model to be saved.
        path (str): The path where the model will be saved.
    """
    model.save(path=path)


if __name__ == "__main__":
    main()
