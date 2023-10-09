from functools import partial
from omegaconf import DictConfig
from optuna.samplers import TPESampler
from optuna.trial import Trial
from typing import Callable

from utils import scorer_pr_auc
from utils import BaseModel

def XGB_params(p: DictConfig, trial: Trial):
    """
    Model Specific parameters

    The purpose of this function is to wrap the hydra config `p` with the
    optuna `Trial` object `trial`. Notice that each model should define their 
    own `params` function.
    """
    return {
        'objective': p.objective,
        'device': p.device,
        'nthread' : p.nthread,
        # default: hist
        'tree_method': p.tree_method,
        'scale_pos_weight' : trial.suggest_int('scale_pos_weight', p.scale_pos_weight.low, p.scale_pos_weight.up),
        'n_estimators' : trial.suggest_int('n_estimators', p.n_estimators.low, p.n_estimators.up),
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

        super().__init__(partial(XGB_params, p), 'xgboost', study_name)

        self.scoring = scoring
        self.direction = direction
        self.cv = cv
        self.sampler = sampler
        self.n_trials = n_trials

    def fit(self, X_train, y_train):
        study = self.optimize(
            X_train,
            y_train,
            self.scoring,
            self.direction,
            self.cv,
            self.sampler,
            self.n_trials,
        )

