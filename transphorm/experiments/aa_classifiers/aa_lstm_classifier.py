import comet_ml
import os
from dataclasses import dataclass
from pathlib import Path

import lightning as L
from comet_ml import Experiment
from comet_ml.integration.pytorch import watch
import torch
from torch import nn
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from transphorm.model_components.data_objects import AATrialDataModule
from transphorm.model_components.model_modules import LSTMClassifer, TrialClassifer
from dotenv import load_dotenv
from torchmetrics import ConfusionMatrix
from comet_ml.integration.pytorch import watch
from IPython import embed


@dataclass
class ExperimentConfigs:
    data_path: str = ""
    experiment_name: str = "unamed"

    epochs: int = 1
    batch_size: int = 32
    learning_rate: int = 1e-4
    input_size: int = 6104
    hidden_size: int = 8
    num_layers: int = 2


def init_comet_logger(experiment_name: str):
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    logger = CometLogger(
        api_key=COMET_API_KEY,
        workspace="transphorm",
        project_name="aa-classifiers",
        experiment_name=experiment_name,
    )
    return logger


def load_data_module(exp_configs: ExperimentConfigs):
    data_mod = AATrialDataModule(
        main_path=exp_configs.data_path,
        batch_size=exp_configs.batch_size,
        num_workers=11,
    )

    data_mod.prepare_data()
    data_mod.setup("fit")
    return data_mod


def build_model(exp_configs: ExperimentConfigs) -> TrialClassifer:
    clf_module = LSTMClassifer(
        input_size=exp_configs.input_size,
        hidden_size=exp_configs.hidden_size,
        num_layers=exp_configs.num_layers,
    )
    model = TrialClassifer(
        model=clf_module,
        optimizer=torch.optim.Adam,
        learning_rate=exp_configs.learning_rate,
    )
    return model


def train(
    model: L.LightningModule,
    logger: CometLogger,
    data_mod: L.LightningDataModule,
    root_dir: Path,
    exp_configs: ExperimentConfigs,
):
    checkpoint_callback = ModelCheckpoint(filename=exp_configs.experiment_name)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=exp_configs.epochs,
        logger=logger,
        default_root_dir=root_dir,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1,
    )
    trainer.fit(model, data_mod)
    return trainer, model


def eval_confusion_matrix(model, data_mod, trainer, logger):
    batch_pred = trainer.predict(model, data_mod)

    y_pred = torch.cat(batch_pred)
    y_test = data_mod.test[:][1]

    num_classes = 2
    confmat = ConfusionMatrix(task="binary", num_classes=num_classes)
    matrix = confmat(y_pred, y_test)
    lables = [f"Class_{i}" for i in range(num_classes)]
    return matrix.numpy(), lables


def main():
    load_dotenv()
    DATA_PATH = Path(os.getenv("TRIAL_DATA_PATH"))
    LOG_DIR = Path(os.getenv("LIGHTNING_LOG_DIR"))

    exp_configs = ExperimentConfigs(
        data_path=DATA_PATH, experiment_name="trial_lstm_clf_v2"
    )

    logger = init_comet_logger(exp_configs.experiment_name)
    logger.log_hyperparams(exp_configs.__dict__)
    model = build_model(exp_configs)
    data_mod = load_data_module(exp_configs)
    logger.log_hyperparams(model.hparams)
    watch(model)

    trainer, model = train(
        model=model,
        logger=logger,
        data_mod=data_mod,
        root_dir=LOG_DIR,
        exp_configs=exp_configs,
    )
    embed()

    confusion_matrix, labels = eval_confusion_matrix(model, data_mod, trainer, logger)
    logger.experiment.log_confusion_matrix(matrix=confusion_matrix, labels=labels)


if __name__ == "__main__":
    main()
