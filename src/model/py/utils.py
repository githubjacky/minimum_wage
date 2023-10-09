from functools import partial
import logging
import numpy as np
from numpy.typing import ArrayLike
import optuna
from optuna.integration.mlflow import MLflowCallback
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_recall_curve, auc
from typing import Callable, List, Optional, Tuple
import xgboost as xgb


class DataSet:
    X_col = [
        'age', 'eduyr', 'work_exp', 'countyname_南投縣', 'countyname_嘉義市',
        'countyname_嘉義縣', 'countyname_基隆市', 'countyname_宜蘭縣',
        'countyname_屏東縣', 'countyname_彰化縣', 'countyname_新竹市',
        'countyname_新竹縣', 'countyname_桃園縣', 'countyname_澎湖縣',
        'countyname_臺中市', 'countyname_臺中縣', 'countyname_臺北市',
        'countyname_臺北縣', 'countyname_臺南市', 'countyname_臺南縣',
        'countyname_臺東縣', 'countyname_花蓮縣', 'countyname_苗栗縣',
        'countyname_雲林縣', 'countyname_高雄市', 'countyname_高雄縣',
        'mar_有配偶(含與人同居)', 'mar_未婚', 'mar_配偶死亡', 'mar_離婚、分居',
        'is_female_yes', 'is_teen_yes', 'is_lths_yes', 'is_lths30_yes',
        'is_hsl_yes', 'is_hsl30_yes'
    ]

    def __init__(
        self,
        data_path: str = 'data/processed/data_v2.csv',
        prediction_stragety: str = 'binary',
        imbalance_strategy: str = 'ignore',
        baking_strategy: str = 'one_hot'
    ) -> None:
        raw = pd.read_csv(data_path)

        cols = [
            'is_female', 'is_teen', 'is_lths', 'is_lths30',
            'is_hsl', 'is_hsl30', 'is_teen', 'group1'
        ]
        pre_processed = pd.get_dummies(raw, columns = cols, drop_first = True)
        cols = ['mar']
        pre_processed = pd.get_dummies(pre_processed, columns = cols)

        match prediction_stragety:
            case 'multi-label':
                pre_processed['group2_factor'] = pd.factorize(pre_processed['group2'])[0]
                self.y_col = ['group2_factor']
            case _:  # binary
                self.y_col = ['group1_treatment']

        match imbalance_strategy:
            case 'under_sampling':
                pass
            case 'over_sampling':
                pass
            case _:  # ignore
                pass

        self.xgb_encode = False
        match baking_strategy:
            case 'ordinal':
                pre_processed['county_factor'] = pd.factorize(pre_processed['countyname'])[0]
            case 'xgb_encode':
                self.xgb_encode = True
            case _:  # one_hot
                 pre_processed = pd.get_dummies(pre_processed, columns = ['countyname'])

        self.df = pre_processed
        self.prediction_stragety = prediction_stragety
        self.imbalance_strategy = imbalance_strategy
        self.baking_strategy = baking_strategy


    def __label_distro(self, x: ArrayLike, tag: str) -> None:
        name, stats = np.unique(x, return_counts = True)
        print(f'label distro({tag}):')
        print(f'{name[0]}: {stats[0]}')
        print(f'{name[1]}: {stats[1]}\n')


    def fetch_train_test(
        self,
        X_col: Optional[List[str]] = None,
        y_col: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 1126,
        verbose: bool = True
    ) -> Tuple[List, List, List, List]:

        Xcol = self.X_col if X_col is None else X_col
        ycol = self.y_col if y_col is None else y_col

        X = self.df[Xcol].to_numpy()
        y = self.df[ycol].to_numpy()


        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size = test_size,
            random_state = random_state,
            stratify = y
        )
        if verbose:
            self.__label_distro(y_train, 'train')
            self.__label_distro(y_test, 'test')

        return X_train, X_test, y_train, y_test


def scorer_pr_auc(estimator, X, y):
    prob = estimator.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, prob)

    return auc(recall, precision)


class BaseModel:
    def __init__(
        self,
        params: Callable,
        model_name: str,
        study_name: str
    ):
        match model_name:
            case 'elnet':
                pass
            case 'rf':
                pass
            case _:  # xgboost
                self.model = lambda trial: xgb.XGBClassifier(**params(trial))

        self.study_name = study_name

        log_dir = Path(f"log/{model_name}")
        log_dir.mkdir(parents = True, exist_ok = True)
        self.log_path = log_dir / f'{study_name}.log'
        self.log_path.unlink(missing_ok = True)

    def __objective(self, trial, X_train, y_train, scoring: str | Callable, cv: int):
        return (
            cross_val_score(
                self.model(trial),
                X_train,
                y_train,
                scoring = scoring,
                cv = cv
            ).mean()
        )

    def optimize(
        self,
        X_train,
        y_train,
        scoring: Callable | str,
        direction: str,
        cv: int,
        sampler,
        n_trials: int,
    ):
        objective = partial(
            self.__objective,
            X_train = X_train,
            y_train = y_train,
            scoring = scoring,
            cv = cv
        )

        metric_name = scoring.__name__ if isinstance(scoring, Callable) else scoring
        mlflc = MLflowCallback(metric_name = metric_name)


        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(self.log_path, mode="w"))
        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()

        study = optuna.create_study(
            direction = direction,
            sampler = sampler,
            study_name = self.study_name
        )

        logger.info("Start optimization.")
        study.optimize(
            objective,
            n_trials,
            callbacks = [mlflc],
        )

        with open(self.log_path) as f:
            assert f.readline().startswith("A new study created")
            assert f.readline() == "Start optimization.\n"

        return study
