import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple


class DataSet:

    X_col = [
        'age', 'work_exp', 'is_female_yes', 'is_lths_yes',
       'is_lths30_yes', 'is_hsl_yes', 'is_hsl30_yes', 'is_teen_yes',
       'edu_college', 'edu_doctoral', 'edu_elementary', 'edu_illiterate',
       'edu_junior_college', 'edu_junior_high', 'edu_master', 'edu_self_study',
       'edu_senior_high_vocational', 'mar_divorced_separated',
       'mar_married_cohabited', 'mar_unmarried', 'mar_widow_widower',
       'countyname_Changhua_County',
       'countyname_Chiayi_City', 'countyname_Chiayi_County',
       'countyname_Gaoxiong_City', 'countyname_Gaoxiong_County',
       'countyname_Hsinchu_City', 'countyname_Hsinchu_County',
       'countyname_Hualian_County', 'countyname_Keelung_City',
       'countyname_Miaoli_County', 'countyname_Nantou_County',
       'countyname_Penghu_County', 'countyname_Pingtung_County',
       'countyname_Taichung_City', 'countyname_Taichung_County',
       'countyname_Taidong_County', 'countyname_Tainan_City',
       'countyname_Tainan_County', 'countyname_Taipei_City',
       'countyname_Taipei_County', 'countyname_Taoyuan_County',
       'countyname_Yilan_County', 'countyname_Yunlin_County'
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
            'is_female', 'is_lths', 'is_lths30',
            'is_hsl', 'is_hsl30', 'is_teen',
        ]
        pre_processed = pd.get_dummies(
            raw,
            columns = cols,
            drop_first = True,
            dtype = float
        )
        pre_processed = pd.get_dummies(
            pre_processed,
            columns = ['edu', 'mar'],
            dtype = float
        )

        match prediction_stragety:
            case 'multi-label':
                pre_processed['group2_factor'] = pd.factorize(pre_processed['group2'])[0]
                self.y_col = ['group2_factor']
            case _:  # binary
                pre_processed = pd.get_dummies(
                    pre_processed,
                    columns = ['group1'],
                    drop_first = True,
                    dtype = int
                )
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
                 pre_processed = pd.get_dummies(
                    pre_processed,
                    columns = ['countyname'],
                    dtype = float
                )

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

        X = self.df[self.X_col if X_col is None else X_col]
        y = self.df[self.y_col if y_col is None else y_col
]
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
