import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import comet_ml
import joblib
import numpy as np
import structlog
from comet_ml.integration.sklearn import log_model
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier

from aa_cls.framework_helpers import (
    dataloader_to_numpy,
    log_test_evaluation,
    log_train_evaluation,
)


def set_hypertune_configs() -> Dict[str, Any]:
    """
    Set the hyperparameter tuning configurations for the Bayesian optimization.

    Returns:
        Dict[str, Any]: A dictionary containing the hyperparameter tuning configurations.
    """
    configs = {
        "algorithm": "bayes",
        "spec": {
            "maxCombo": 30,
            "objective": "maximize",
            "metric": "test_balanced_accuracy",
            "minSampleSize": 700,
            "retryLimit": 20,
            "retryAssignLimit": 0,
        },
        "parameters": {
            "epochs": [500, 1000, 1500, 2000],
            "dropout": [0.2, 0.4, 0.6, 0.8],
            "kernel_sizes": ["32, 15, 9", "16, 10, 6", "8, 5, 3"],
            "filter_sizes": ["128, 256, 128", "64, 128, 64"],
            "lstm_size": [2, 4, 6, 8, 10],
            "random_state": [42],
        },
    }
    return configs


def exeperiment_configs(project_name: str) -> Dict[str, Any]:
    """
    Set the experiment configurations for Comet.ml logging.

    Args:
        project_name (str): The name of the project.

    Returns:
        Dict[str, Any]: A dictionary containing the experiment configurations.
    """
    exp_configs = {
        "project_name": project_name,
        "auto_param_logging": True,
        "auto_metric_logging": True,
        "auto_histogram_weight_logging": True,
        "auto_histogram_gradient_logging": True,
        "auto_histogram_activation_logging": True,
        "display_summary_level": 0,
    }
    return exp_configs


def load_data(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split the data from the given path.

    Args:
        path (Path): The path to the data file.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    X, y = dataloader_to_numpy(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test


def train(
    exp: comet_ml.Experiment, X_train: np.ndarray, y_train: np.ndarray
) -> LSTMFCNClassifier:
    """
    Train the LSTMFCNClassifier model with the given experiment parameters.

    Args:
        exp (comet_ml.Experiment): The Comet.ml experiment object.
        X_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.

    Returns:
        LSTMFCNClassifier: The trained model.
    """
    kernel_sizes = tuple(map(int, exp.get_parameter("kernel_sizes").split(",")))
    filter_sizes = tuple(map(int, exp.get_parameter("filter_sizes").split(",")))
    classes = np.unique(y_train)

    model_params = {
        "dropout": exp.get_parameter("dropout"),
        "kernel_sizes": kernel_sizes,
        "filter_sizes": filter_sizes,
        "lstm_size": exp.get_parameter("lstm_size"),
        "random_state": exp.get_parameter("random_state"),
    }

    model = LSTMFCNClassifier(**model_params)
    model.fit(X_train, y_train)

    return model


def run_optimizer(
    project_name: str,
    opt: comet_ml.Optimizer,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    log: structlog.BoundLogger,
    model_save_dir: Path,
) -> None:
    """
    Run the Bayesian optimization process for hyperparameter tuning.

    Args:
        project_name (str): The name of the project.
        opt (comet_ml.Optimizer): The Comet.ml optimizer object.
        X_train (np.ndarray): The training features.
        X_test (np.ndarray): The testing features.
        y_train (np.ndarray): The training labels.
        y_test (np.ndarray): The testing labels.
        log (structlog.BoundLogger): The logger object.
        model_save_dir (Path): The directory to save the trained models.
    """
    exp_configs = exeperiment_configs(project_name)
    for exp in opt.get_experiments(**exp_configs):
        model = train(exp, X_train, y_train)
        params = model.get_params()
        exp.log_parameters(params)

        train_pred = model.predict(X_train)
        train_pred_prob = model.predict_proba(X_train)[:, 1]
        log_train_evaluation(
            y=y_train,
            y_pred=train_pred,
            y_pred_prob=train_pred_prob,
            exp=exp,
        )

        test_pred = model.predict(X_test)
        test_pred_prob = model.predict_proba(X_test)[:, 1]
        log_test_evaluation(
            y=y_test,
            y_pred=test_pred,
            y_pred_prob=test_pred_prob,
            data_cat="test",
            exp=exp,
        )

        joblib.dump(model, model_save_dir / f"{exp.name}.joblib")
        log_model(
            experiment=exp, model_name=exp.name, model=model, persistence_module=pickle
        )
        exp.end()


def main() -> None:
    """
    The main function to run the LSTM-FCN classifier with Bayesian hyperparameter tuning.
    """
    load_dotenv()
    log = structlog.get_logger()
    PROJECT_NAME = "lstmnfcn_bayes_tuning_5_day_weighted"
    MODEL_SAVE_DIR = Path("/projects/p31961/aa_cls/models/aa_classifiers/sk_models")

    DATA_PATH = Path(os.getenv("DATA_PATH_5_DAY"))
    COMET_API_KEY = os.getenv("COMET_API_KEY")

    log.info("loading data")
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    log.info("configuring optimizer")
    opt = comet_ml.Optimizer(config=set_hypertune_configs())

    run_optimizer(
        project_name=PROJECT_NAME,
        opt=opt,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        log=log,
        model_save_dir=MODEL_SAVE_DIR,
    )
    log.info("experiment complete")


if __name__ == "__main__":
    main()
