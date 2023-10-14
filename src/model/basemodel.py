from functools import partial
import logging
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.integration.mlflow import MLflowCallback
from omegaconf import DictConfig
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from typing import Dict


class BaseModel:
    def __init__(
        self,
        model_name: str,
        study_name: str,
        cfg: DictConfig,
        fixed_param: Dict
    ):

        self.study_name = study_name

        log_dir = Path(f"log/{model_name}")
        log_dir.mkdir(parents = True, exist_ok = True)
        self.log_path = log_dir / f'{study_name}.log'
        self.log_path.unlink(missing_ok = True)

        self.cfg = cfg
        self.fixed_param = fixed_param

    def objective(self, trial, _X_train, _y_train):
        # model specific optuna_param
        params = self.fixed_param | self.optuna_param(trial)
        kf = StratifiedKFold(
            self.cfg.tune.cv,
            shuffle = True,
            random_state = self.cfg.preprocess.random_state
        )
        val_eval_metrics = []
        for train_idx, val_idx in kf.split(
            _X_train.values.tolist(),
            _y_train.values.tolist()
        ):
            X_train = _X_train.iloc[train_idx]
            X_val = _X_train.iloc[val_idx]
            y_train = _y_train.iloc[train_idx]
            y_val = _y_train.iloc[val_idx]

            # model specific train
            val_eval_metrics.append(
                self.train(params, X_train, X_val, y_train, y_val, 'val')[0]
            )

        return np.mean(val_eval_metrics)

    def fit(self, X_train, y_train):
        # the objective function should implemented with model spefic class
        objective = partial(
            self.objective,
            _X_train = X_train,
            _y_train = y_train,
        )
        mlflc = MLflowCallback(
            tracking_uri = 'https://dagshub.com/githubjacky/minimum_wage.mlflow',
            metric_name = self.cfg.model.eval_metric
        )

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(self.log_path, mode = "w"))
        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()

        study = optuna.create_study(
            direction = self.cfg.model.eval_metric_direction,
            sampler = eval(self.cfg.tune.sampler)(),
            study_name = 'tune_' + self.study_name
        )
        logger.info("Start optimization.")
        study.optimize(
            objective,
            self.cfg.tune.n_trials,
            callbacks = [mlflc],
        )
        with open(self.log_path) as f:
            assert f.readline().startswith("A new study created")
            assert f.readline() == "Start optimization.\n"
        self.study = study

        best_model_param = self.fixed_param | self.study.best_params
        self.best_param_str = {
            k: str(v)
            for k, v in best_model_param.items()
        }

        return best_model_param


    def log_test(self, y_test, pred) -> None:
        metric_dict = classification_report(y_test, pred, output_dict = True)

        mlflow.log_metric('precision_1', metric_dict['1']['precision'])
        mlflow.log_metric('recall_1', metric_dict['1']['recall'])
        mlflow.log_metric('f1_1', metric_dict['1']['f1-score'])
        mlflow.log_metric('support_1', metric_dict['1']['support'])
        mlflow.log_metric('precision_0', metric_dict['0']['precision'])
        mlflow.log_metric('recall_0', metric_dict['0']['recall'])
        mlflow.log_metric('f1_0', metric_dict['0']['f1-score'])
        mlflow.log_metric('support_0', metric_dict['0']['support'])
        mlflow.log_metric('micro_f1', metric_dict['accuracy'])
        mlflow.log_metric('macro_f1', metric_dict['macro avg']['f1-score'])
        mlflow.log_metric('weighted_f1', metric_dict['weighted avg']['f1-score'])

        cm = confusion_matrix(y_test, pred, normalize = 'pred')
        disp = ConfusionMatrixDisplay(confusion_matrix = cm)
        disp.plot()
        mlflow.log_figure(disp.figure_, 'confustion_matrix.png')
