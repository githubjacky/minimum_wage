from omegaconf import DictConfig
from optuna.trial import Trial
import mlflow
import matplotlib.pyplot as plt
from basemodel import BaseModel
import pandas as pd
from typing import Dict
import xgboost

class XGB(BaseModel):
    def __init__(
        self,
        cfg: DictConfig,
        study_name: str
    ):
        study_name = f'xgb_{study_name}'
        fixed_param = {
            'objective': cfg.model.objective,
            'device': cfg.model.device,
            'nthread' : cfg.model.nthread,
            # default: hist
            'tree_method': cfg.model.tree_method,
            'eval_metric': cfg.model.eval_metric,
        }

        super().__init__(
            'xgb',
            study_name,
            cfg,
            fixed_param
        )
        self.es = xgboost.callback.EarlyStopping(
            rounds = self.cfg.model.early_stopping_rounds,
            metric_name = self.cfg.model.eval_metric,
            maximize = True if self.cfg.model.eval_metric_direction == 'maximize' else False,
            save_best = True,
            min_delta = self.cfg.model.min_delta
        )

    def optuna_param(self, trial: Trial):
        p = self.cfg.model
        return {
            'scale_pos_weight' : trial.suggest_int('scale_pos_weight', p.scale_pos_weight.low, p.scale_pos_weight.up),
            # default: 6
            'max_depth':trial.suggest_int('max_depth', p.max_depth.low, p.max_depth.up),
            # default: 1
            'min_child_weight':trial.suggest_int('min_child_weight', p.min_child_weight.low, p.min_child_weight.up),
            # default: 0.3
            'learning_rate':trial.suggest_float('learning_rate', p.learning_rate.low, p.learning_rate.up, log = p.learning_rate.log),
            # default: 1
            "subsample": trial.suggest_float("subsample", p.subsample.low, p.subsample.up, step = p.subsample.step),
            # default: 1.
            'colsample_bylevel':trial.suggest_float('colsample_bylevel', p.colsample_bylevel.low, p.colsample_bylevel.up, step = p.colsample_bylevel.step),
            # default: 0.
            'reg_alpha':trial.suggest_float('reg_alpha', p.reg_alpha.low, p.reg_alpha.up),
            # default: 1.
            'reg_lambda':trial.suggest_float('reg_lambda', p.reg_lambda.low, p.reg_lambda.up),
            # default: 0.
            'min_split_loss':trial.suggest_float('min_split_loss', p.gamma.low, p.gamma.up)
        }

    def train(self, params, X_train, X_test, y_train, y_test, tag: str = 'test'):
        dtrain = xgboost.DMatrix(X_train, label = y_train)
        dtest = xgboost.DMatrix(X_test, label = y_test)

        evals = [(dtrain, 'train'), (dtest, tag)]
        evals_result = {}
        bst = xgboost.train(
            params = params,
            dtrain = dtrain,
            num_boost_round = 2000,  # large enough for trigger early stopping
            evals = evals,
            verbose_eval = self.cfg.model.verbose_eval,
            evals_result = evals_result,
            early_stopping_rounds = self.cfg.model.early_stopping_rounds,
            maximize = True if self.cfg.model.eval_metric_direction == 'maximize' else False,
            # callbacks = [self.es]
        )

        return bst.best_score, bst, evals_result

    def log_eval_metric(self, evals_result: Dict) -> None:
        train_eval_result = evals_result['train'][self.cfg.model.eval_metric]
        test_eval_result = evals_result['test'][self.cfg.model.eval_metric]

        fig = plt.figure()
        plt.plot(train_eval_result, label = 'train')
        plt.plot(test_eval_result, label = 'test')
        plt.xlabel('iterations')
        plt.ylabel(self.cfg.model.eval_metric)
        plt.title('')
        plt.legend()
        mlflow.log_figure(fig, 'fit_result.png')

    def test(self, best_model_param, X_train, X_test, y_train, y_test):
        strategy_list = self.study_name.split('_')
        extra_tags = {
            'prediction_strategy': strategy_list[1],
            'imbalance_strategy': strategy_list[2],
            'baking_strategy': strategy_list[3],
            'ouptuna_sampler': strategy_list[4],
            'eval_metric': strategy_list[5]
        }

        exper = mlflow.set_experiment('test')
        with mlflow.start_run(
            experiment_id = exper.experiment_id,
            run_name = self.study_name
        ):
            mlflow.set_tags(extra_tags)
            mlflow.xgboost.autolog(
                log_input_examples = True,
                log_model_signatures = True,
                log_models = True,
                log_datasets = False,
                exclusive = False,
                silent = True,
                registered_model_name = self.study_name,
            )
            _, best_model, evals_result = self.train(
                best_model_param,
                X_train,
                X_test,
                y_train,
                y_test
            )
            pred = [
                1 if i >= 0.5 else 0
                for i in best_model.predict(xgboost.DMatrix(X_test, label = y_test))
            ]
            self.log_eval_metric(evals_result)
            self.log_test(y_test, pred)
