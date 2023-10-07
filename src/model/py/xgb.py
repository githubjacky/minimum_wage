from functools import partial
from omegaconf import DictConfig
from optuna.samplers import TPESampler
from typing import Callable

from utils import scorer_pr_auc
from utils import BaseModel

def params(p: DictConfig, trial):
    return {
        'objective': p.objective,
        'tree_method': p.tree_method,
        'device': p.device,
        'nthread' : p.nthread,
        "n_estimators" : trial.suggest_int('n_estimators', p.n_estimators.low, p.n_estimators.up),
        'max_depth':trial.suggest_int('max_depth', p.max_depth.low, p.max_depth.up),
        'min_child_weight':trial.suggest_int('min_child_weight', p.min_child_weight.low, p.min_child_weight.up),
        'learning_rate':trial.suggest_float('learning_rate', p.learning_rate.low, p.learning_rate.up, log = p.learning_rate.log),
        "subsample": trial.suggest_float("subsample", p.subsample.low, p.subsample.up),
        'colsample_bytree':trial.suggest_float('colsample_bytree', p.colsample_bytree.low, p.colsample_bytree.up, step = p.colsample_bytree.step),
        'reg_alpha':trial.suggest_int('reg_alpha', p.reg_alpha.low, p.reg_alpha.up),
        'reg_lambda':trial.suggest_int('reg_lambda', p.reg_lambda.low, p.reg_lambda.up),
        'gamma':trial.suggest_float('gamma', p.gamma.low, p.gamma.up, step = p.gamma.step)
    }

class XGB(BaseModel):
    def __init__(
        self,
        p: DictConfig,
        study_name: str,
        scoring: Callable | str = scorer_pr_auc,
        direction: str = 'maximize',
        cv: int = 5,
        sampler = TPESampler(),
        n_trials: int = 2000,
    ):
        study_name += '_xgboost'

        super().__init__(partial(params, p), 'xgboost', study_name)

        self.scoring = scoring
        self.direction = direction
        self.cv = cv
        self.sampler = sampler
        self.n_trials = n_trials

    def fit(self, X_train, y_train, verbose):
        study = self.optimize(
            X_train,
            y_train,
            self.scoring,
            self.direction,
            self.cv,
            self.sampler,
            self.n_trials,
            verbose
        )

