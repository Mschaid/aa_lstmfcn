from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    auc,
    accuracy_score,
    balanced_accuracy_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from transphorm.model_components.data_objects import AATrialDataModule


def load_py_data_to_np(path: Path):
    data = torch.load(path)
    data_no_na = data[~torch.isnan(data[:, 0])]
    data_np = data_no_na.detach().numpy()
    return data_np


def split_data_reproduce(
    X, y: np.array, train_size=0.7, test_size=0.5, random_state: int = 42
):

    X_train, X_, y_train, y_ = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# convert x to tensor
def log_evaluaton(y, y_pred, y_pred_prob, data_cat, exp):

    conf_mat = confusion_matrix(y, y_pred)

    exp.log_confusion_matrix(
        matrix=conf_mat,
        labels=["Avoid", "Escape"],
        title=f"{data_cat} Confusion Matrix",
    )

    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    exp.log_curve(f"{data_cat} ROC Curve", x=fpr, y=tpr)

    evals = {
        "f1_score": f1_score(y, y_pred),
        "f1_score_weighted": f1_score(y, y_pred, average="weighted"),
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision_weighted": precision_score(y, y_pred, average="weighted"),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "recall_weighted": recall_score(y, y_pred, average="weighted"),
        "roc_auc": auc(fpr, tpr),
    }

    for k, v in evals.items():
        exp.log_metric(k, v)


def log_train_evaluation(y, y_pred, y_pred_prob, exp):
    data_cat = "train"

    conf_mat = confusion_matrix(y, y_pred)

    exp.log_confusion_matrix(
        matrix=conf_mat,
        labels=["Avoid", "Escape"],
        title=f"{data_cat} Confusion Matrix",
    )

    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    exp.log_curve(f"{data_cat} ROC Curve", x=fpr, y=tpr)

    train_evals = {
        f"{data_cat}_f1_score": f1_score(y, y_pred),
        f"{data_cat}_f1_score_weighted": f1_score(y, y_pred, average="weighted"),
        f"{data_cat}_accuracy": accuracy_score(y, y_pred),
        f"{data_cat}_balanced_accuracy": balanced_accuracy_score(y, y_pred),
        f"{data_cat}_precision_weighted": precision_score(
            y, y_pred, average="weighted"
        ),
        f"{data_cat}_precision": precision_score(y, y_pred),
        f"{data_cat}_recall": recall_score(y, y_pred),
        f"{data_cat}_recall_weighted": recall_score(y, y_pred, average="weighted"),
        f"{data_cat}_roc_auc": auc(fpr, tpr),
    }

    for k, v in train_evals.items():
        exp.log_metric(k, v)


def log_test_evaluation(y, y_pred, y_pred_prob, exp):
    data_cat = "test"

    conf_mat = confusion_matrix(y, y_pred)

    exp.log_confusion_matrix(
        matrix=conf_mat,
        labels=["Avoid", "Escape"],
        title=f"{data_cat} Confusion Matrix",
    )

    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    exp.log_curve(f"{data_cat} ROC Curve", x=fpr, y=tpr)

    test_evals = {
        f"{data_cat}_f1_score": f1_score(y, y_pred),
        f"{data_cat}_f1_score_weighted": f1_score(y, y_pred, average="weighted"),
        f"{data_cat}_accuracy": accuracy_score(y, y_pred),
        f"{data_cat}_balanced_accuracy": balanced_accuracy_score(y, y_pred),
        f"{data_cat}_precision_weighted": precision_score(
            y, y_pred, average="weighted"
        ),
        f"{data_cat}_precision": precision_score(y, y_pred),
        f"{data_cat}_recall": recall_score(y, y_pred),
        f"{data_cat}_recall_weighted": recall_score(y, y_pred, average="weighted"),
        f"{data_cat}_roc_auc": auc(fpr, tpr),
    }

    for k, v in test_evals.items():
        exp.log_metric(k, v)


# comment
def dataloader_to_numpy(path: Path, data_loader=AATrialDataModule):
    data = data_loader(path)
    data.prepare_data()
    X = data.data[:][0].detach().numpy()
    y = data.data[:][1].detach().numpy()
    return X, y
