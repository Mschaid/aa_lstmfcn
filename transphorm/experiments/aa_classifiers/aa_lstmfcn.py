from pathlib import Path
import pickle
import comet_ml
from dotenv import load_dotenv
from transphorm.model_components.data_objects import AATrialDataModule
from transphorm.framework_helpers import (
    dataloader_to_numpy,
    split_data_reproduce,
    evaluate,
    setup_comet_experimet,
)
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
import os
from sklearn.metrics import confusion_matrix

import joblib


def build_model():
    lstmfcn = LSTMFCNClassifier(n_epochs=2000, dropout=0.2)
    return lstmfcn


def main():
    MODEL_SAVE_DIR = Path(
        "/Users/mds8301/Development/transphorm/models/sk/aa_classifiers"
    )

    DATA_PATH = Path(os.getenv("TRIAL_DATA_PATH"))
    EXPERIMENT_NAME = "lstmfcn_aa_trial_v0"
    COMET_API_KEY = os.getenv("COMET_API_KEY")

    MODEL_SAVE_PATH = MODEL_SAVE_DIR / f"{EXPERIMENT_NAME}.pkl"
    # set up comet experiment
    exp = comet_ml.Experiment(
        api_key=COMET_API_KEY, workspace="transphorm", project_name="aa-classifiers"
    )

    # load data from pytroch loaders
    X, y = dataloader_to_numpy(DATA_PATH)

    # spit data with reproduce
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_reproduce(X, y)

    # define lstmfcn model
    model = build_model()

    # train model
    model.fit(X_train, y_train)

    # log training metrics
    y_train_pred = model.predict(X_train)
    with exp.train():
        metrics = evaluate(y_train, y_train_pred)
        exp.log_metrics(metrics)

    # log testing metrics,
    y_test_pred = model.predict(X_test)
    with exp.test():
        metrics = evaluate(y_test, y_test_pred)
        exp.log_metrics(metrics)

    # write confusion mat
    confmat = confusion_matrix(y_pred, y_test)
    lables = ["avoid", "escape"]
    exp.log_confusion_matrix(confmat, lables)

    # save model
    joblib.dump(model, MODEL_SAVE_PATH)
    exp.log_model(EXPERIMENT_NAME, MODEL_SAVE_PATH.as_posix())
    exp.end()


if __name__ == "__main__":
    main()
