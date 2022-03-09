import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


feature_columns = ['moran', 'no2', 'Level_0', 'Level_1', 'Level_2', 'Level_3', 'Level_4', 'Level_5',
                  'Sector_0', 'Sector_1', 'Sector_2', 'Sector_3',
                  'wind_spd', 'wd_sin', 'wd_cos', 'Ship_length', 'ship_spd_avg']


def model_dict(class_weight_dict):
    models_dict = {'RandomForest': dict(model=RandomForestClassifier(random_state=0, oob_score=True, n_jobs=3,
                                                                     verbose=2,
                                                                     class_weight=class_weight_dict, n_estimators=500),
                                        param_dict=dict(min_samples_leaf=np.arange(2, 36, 2),
                                                        max_features=('sqrt', 0.4, 0.5),
                                                        criterion=('gini', 'entropy'))),
                   'SVM': dict(model=SVC(random_state=0, kernel='rbf', gamma='scale', verbose=2,
                                         class_weight=class_weight_dict),
                               param_dict=dict(C=(2.0e-2, 0.5e-1, 1.0e-1, 1.5e-1,
                                                  2.0e-1, 2.5e-1, 2.0))),
                   'Linear_SVM': dict(model=LinearSVC(random_state=0, dual=False, verbose=2,
                                                      class_weight=class_weight_dict),
                                      param_dict={'C': (2.0e-2, 2.0e-1, 2.0, 2.0e1, 2.0e2)
                                                  }),
                   'Logistic': {'model': LogisticRegression(random_state=0, solver='saga', n_jobs=3, verbose=2,
                                                            l1_ratio=0.5, class_weight=class_weight_dict),
                                'param_dict': dict(penalty=('l1', 'l2', 'elasticnet', 'none'),
                                                   C=(0.0001, 0.001, 0.1, 1),
                                                   max_iter=(100, 120, 150))},
                   'XGB': {'model': xgb.XGBClassifier(objective="binary:logistic", eval_metric='aucpr',
                                                      random_state=0, n_estimators=500, booster='gbtree', n_jobs=3),

                           'param_dict': dict(gamma=np.arange(0.05, 0.5, 0.05), max_depth=(2, 3, 5, 6),
                                              min_child_weight=(2, 4, 6, 8, 10, 12),
                                              subsample=np.arange(0.6, 1.0, 0.1),
                                              colsample_bytree=np.arange(0.6, 1.0, 0.1),
                                              colsample_bylevel=np.arange(0.6, 1.0, 0.1),
                                              learning_rate=(0.001, 0.01, 0.1, 0.2, 0.3, 0.4),
                                              reg_alpha=(0, 1.0e-5, 5.0e-4, 1.0e-3, 1.0e-2, 0.1, 1))
                           }}
    return models_dict