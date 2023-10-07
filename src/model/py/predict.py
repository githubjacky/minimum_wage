import hydra
from omegaconf import DictConfig
from optuna.samplers import TPESampler

from utils import DataSet, scorer_pr_auc
from xgb import XGB


@hydra.main(config_path="../../../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    data = DataSet(cfg.preprocess.raw_path)
    X_train, X_test, y_train, y_test = data.fetch_train_test(
        cfg.preprocess.X_col,
        cfg.preprocess.y_col,
        cfg.preprocess.test_size,
        cfg.preprocess.random_state,
        cfg.preprocess.verbose
    )
    study_name = '_'.join((
        cfg.preprocess.prediction_strategy,
        cfg.preprocess.imbalance_strategy,
        cfg.preprocess.baking_strategy,
        cfg.tuning.scoring
    ))
    scoring = (
        eval(cfg.tuning.scoring)
        if cfg.tuning.scoring == 'scorer_pr_auc'
        else
        cfg.tuning.scoring
    )
    estimator = eval(cfg.model.estimator)(
        cfg.model,
        study_name,
        scoring,
        cfg.tuning.direction,
        cfg.tuning.cv,
        eval(cfg.tuning.sampler)(),
        cfg.tuning.n_trials
    )
    estimator.fit(X_train, y_train, cfg.tuning.verbose)


if __name__ == "__main__":
    main()
