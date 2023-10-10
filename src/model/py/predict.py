import hydra
from omegaconf import DictConfig

from utils import DataSet
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
        cfg.tune.sampler,
        cfg.model.eval_metric
    ))

    estimator = eval(cfg.model.estimator)(cfg, study_name)
    best_model_param = estimator.fit(X_train, y_train)

    if cfg.mode.test:
        estimator.test(best_model_param, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
