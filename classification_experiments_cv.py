import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.model_selection import RandomizedSearchCV,  GroupKFold
from model_params import model_dict, feature_columns

np.random.seed(1234)


def do_train_test_split(labeled_df, cv_index, test_size=0.2):

    df_model = labeled_df[labeled_df.sector == 1].reset_index(drop=True)
    number_of_plumes = len(df_model.Plume_id.unique())
    test_ids = np.random.choice(df_model.Plume_id.unique(), int(number_of_plumes * test_size))
    test = df_model[df_model.Plume_id.isin(test_ids)].reset_index(drop=True)
    train = df_model[~df_model.Plume_id.isin(test_ids)].reset_index(drop=True)
    ds_path = f'./data_sets'
    if not os.path.exists(ds_path):
        os.mkdir(ds_path)
    train.to_csv(f'./{ds_path}/train_set_cv_{cv_index}.csv')
    test.to_csv(f'./{ds_path}/test_set_cv_{cv_index}.csv')
    weight_for_0 = (1 / len(train[train.Label==0])) * (len(train) / 2.0)
    weight_for_1 = (1 / len(train[train.Label==1])) * (len(train) / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return train, test, class_weight

def threshold_optimization(train_set, gkf, cv_index):

    qs = np.arange(0, 0.98, 0.05)
    cvi = 0
    df_cv = pd.DataFrame()
    for tr_ind, val_ind in gkf.split(train_set, groups=train_set['Plume_id']):
        train = train_set.iloc[tr_ind].reset_index(drop=True)
        val = train_set.iloc[val_ind].reset_index(drop=True)
        df_run = pd.DataFrame()
        for i in range(len(qs)):
            m_high_quantile = np.quantile(train.moran_high, qs[i])
            m_quantile = np.quantile(train.moran, qs[i])
            no2_quantile = np.quantile(train.no2, qs[i])
            pred_m_high = np.where(val.moran_high >= m_high_quantile, 1, 0)
            pred_m = np.where(val.moran >= m_quantile, 1, 0)
            pred_no2 = np.where(val.no2 >= no2_quantile, 1, 0)
            f1_m_high = metrics.f1_score(val['Label'].values.reshape(-1, 1),
                                         pred_m_high, pos_label=1, average='binary')
            f1_m = metrics.f1_score(val['Label'].values.reshape(-1, 1),
                                    pred_m, pos_label=1, average='binary')
            f1_no2 = metrics.f1_score(val['Label'].values.reshape(-1, 1),
                                      pred_no2, pos_label=1, average='binary')
            d = {'Qs': qs[i],
                 ' m_high_quantile': m_high_quantile,
                 'm_quantile': m_quantile,
                 'no2_quantile': no2_quantile,
                 'f1_m_high': f1_m_high,
                 'f1_m': f1_m,
                 'f1_no2': f1_no2,
                 'cv_index': cvi}
            df_run = df_run.append(d, ignore_index=True)
        df_cv = df_cv.append(df_run, ignore_index=True)
        cvi += 1
    df_cv_gr = df_cv.groupby('Qs', as_index=False).mean()
    df_cv_gr.to_csv(f'./clf_results/threshold_gs_{cv_index}.csv')
    argmax_m = df_cv_gr['f1_m'].argmax()
    argmax_m_high = df_cv_gr['f1_m_high'].argmax()
    argmax_no2 = df_cv_gr['f1_no2'].argmax()
    d_thr = {'Q_moran': df_cv_gr['Qs'][argmax_m],
             'Q_moran_high': df_cv_gr['Qs'][argmax_m_high],
             'Q_no2': df_cv_gr['Qs'][argmax_no2],
             'Moran_thr': np.quantile(train_set.moran, df_cv_gr['Qs'][argmax_m]),
             'Moran_high_thr': np.quantile(train_set.moran_high, df_cv_gr['Qs'][argmax_m_high]),
             'NO2_thr': np.quantile(train_set.no2, df_cv_gr['Qs'][argmax_no2])}
    return pd.DataFrame(d_thr, index=[0])

def cross_validation(train_set, class_weight,  feature_columns, cv_index):

    models_dict = model_dict(class_weight)
    best_model_dict = dict.fromkeys(models_dict.keys())
    new_path = f'./clf_results'
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    model_path = f'./best_models'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    gkf = GroupKFold(n_splits=5)
    n_splits = gkf.get_n_splits(groups=train_set['Plume_id'])
    for model_name in models_dict.keys():
        best_model_dict[model_name] = {'model': None,
                                       'param_dict': None,
                                       'coefs': None}
        clf = models_dict[model_name]['model']
        params = models_dict[model_name]['param_dict']
        gs = RandomizedSearchCV(clf, params, cv=n_splits, random_state=0,
                                n_iter=60, n_jobs=3, error_score='raise', scoring='average_precision')
        scaler = preprocessing.StandardScaler()
        gs.fit(scaler.fit_transform(train_set[feature_columns].values),
               train_set.Label.values.reshape(-1,1))
        with open(f'./{model_path}/scaler.pkl', 'wb') as fid:
            pickle.dump(scaler, fid)
            fid.close()
        pd.DataFrame(gs.cv_results_).to_csv(f'./clf_results/{model_name}_cv_results_cv_index_{cv_index}.csv')
        best_model_dict[model_name]['model'] = gs.best_estimator_
        best_model_dict[model_name]['param_dict'] = gs.best_params_
     # Threshold optimization
    df_thr = threshold_optimization(train_set, gkf, cv_index)
    df_thr.to_csv(f'./clf_results/threshold_cv_index_{cv_index}.csv')
    return best_model_dict, scaler, df_thr




def quality_metrics(test_set, best_model_dict, feature_columns, scaler, cv_index, df_thr):
    prec_recall_path = f'./clf_results/prec_recalls'
    if not os.path.exists(prec_recall_path):
        os.mkdir(prec_recall_path)
    df = pd.DataFrame(columns=['Model', 'Prec', 'Recall', 'F1', 'AP', 'Balanced_Acc', 'AUC'])
    models = list(best_model_dict.keys())
    # Add thereshold-based models to the list of reported models
    models.append('Moran')
    models.append('Moran_high')
    models.append('NO2')
    df['Model'] = models
    plt.figure(figsize=(12, 6))
    for model_name in best_model_dict.keys():
        df_metrices = pd.DataFrame()
        if model_name in ['Linear_SVM', 'SVM', 'Logistic']:
            test_set[f'y_score_{model_name}'] = best_model_dict[model_name]['model'].decision_function(scaler.transform(test_set[feature_columns].values))
        else:
            test_set[f'y_score_{model_name}'] = best_model_dict[model_name]['model'].predict_proba(scaler.transform(test_set[feature_columns].values))[:,1]
        prec_, recall_, _ = metrics.precision_recall_curve(test_set['Label'].values.reshape(-1, 1),
                                                           test_set[f'y_score_{model_name}'].values.reshape(-1,1),
                                                           pos_label=best_model_dict[model_name]['model'].classes_[1])
        df_metrices[f'prec_{model_name}'] = prec_
        df_metrices[f'recall_{model_name}'] = recall_
        df_metrices.to_csv(f'{prec_recall_path}/prec_recall_{model_name}_cv_{cv_index}.csv')
        test_set[f'y_pred_{model_name}'] = best_model_dict[model_name]['model'].predict(scaler.transform(test_set[feature_columns].values))
        auc_score = metrics.roc_auc_score(test_set['Label'].values.reshape(-1, 1),
                                          test_set[f'y_score_{model_name}'].values.reshape(-1,1), average='weighted').round(3)
        df.loc[df.Model == model_name, 'Prec'] = metrics.precision_score(test_set['Label'].values.reshape(-1,1),
                                                                         test_set[f'y_pred_{model_name}'].values.reshape(-1,1)).round(3)
        df.loc[df.Model == model_name, 'Recall'] = metrics.recall_score(test_set['Label'].values.reshape(-1,1),
                                                                        test_set[f'y_pred_{model_name}'].values.reshape(-1,1)).round(3)
        df.loc[df.Model == model_name, 'F1'] = metrics.f1_score(test_set['Label'].values.reshape(-1,1),
                                                                test_set[f'y_pred_{model_name}'].values.reshape(-1, 1),
                                                                average='binary').round(3)
        df.loc[df.Model == model_name, 'AP'] = metrics.average_precision_score(test_set['Label'].values.reshape(-1, 1),
                                                   test_set[f'y_score_{model_name}'].values.reshape(-1,1), average='weighted').round(3)
        df.loc[df.Model == model_name, 'Balanced_Acc'] = metrics.balanced_accuracy_score(test_set['Label'].values.reshape(-1,1),
                                                                                         test_set[ f'y_pred_{model_name}'].values.reshape(
                                                                                             -1, 1)).round(3)
        df.loc[df.Model == model_name, 'AUC'] = auc_score

        with open(f'./clf_results/Best_params_{model_name}_cv_{cv_index}.txt', 'w') as f:
            print(best_model_dict[model_name]['param_dict'], file=f)
            f.close()
    # Generation of the results for the threshold-based models
    thr_names = ['Moran', 'Moran_high', 'NO2']
    var_names = ['moran', 'moran_high', 'no2']
    for ind in range(len(thr_names)):
        thr_type = thr_names[ind]
        v_n = var_names[ind]
        test_set[f'y_pred_{thr_type}'] = np.where(test_set[v_n] < df_thr[f'{thr_type}_thr'].values[0], 0, 1)
        df.loc[df.Model == thr_type, 'Prec'] = metrics.precision_score(test_set['Label'].values.reshape(-1,1),
                                                                       test_set[f'y_pred_{thr_type}'].values.reshape(-1,1)).round(3)
        df.loc[df.Model == thr_type, 'Recall'] = metrics.recall_score(test_set['Label'].values.reshape(-1,1),
                                                                      test_set[f'y_pred_{thr_type}'].values.reshape(-1,1)).round(3)
        df.loc[df.Model == thr_type, 'F1'] = metrics.f1_score(test_set['Label'].values.reshape(-1, 1),
                                                              test_set[f'y_pred_{thr_type}'].values.reshape(-1, 1),
                                                              average='binary').round(3)
        df.loc[df.Model == thr_type, 'Balanced_Acc'] = metrics.balanced_accuracy_score(test_set['Label'].values.reshape(-1, 1),
                                                                                       test_set[f'y_pred_{thr_type}'].values.reshape(-1, 1)).round(3)
        ap_score = metrics.average_precision_score(test_set['Label'].values.reshape(-1, 1), test_set[v_n].values.reshape(-1, 1),
                                                   average='weighted').round(3)
        df.loc[df.Model == thr_type, 'AP'] = ap_score
    return df, test_set


def classify_pixels(feature_columns, cv=5):

    labeled_df = pd.read_csv('data/labeled_data.csv')
    results_df = pd.DataFrame()
    test_sets = pd.DataFrame()
    for i in range(cv):
        train_set, test_set, class_weight_dict = do_train_test_split(labeled_df, cv_index=i)
        #Grid search for a given test set
        best_model_dict, scaler, df_thr = cross_validation(train_set, class_weight_dict, feature_columns, cv_index=i)
        clf_results, test_set = quality_metrics(test_set, best_model_dict, feature_columns, scaler, cv_index=i, df_thr=df_thr)
        clf_results['cv_index'] = i
        test_set['cv_index'] = i
        results_df = results_df.append(clf_results, ignore_index=True)
        test_sets = test_sets.append(test_set, ignore_index=True)
    test_sets.to_csv(f'./clf_results/all_test_sets_with_clf_results.csv')
    results_df.to_csv(f'./clf_results/results_df.csv')

if __name__ == '__main__':
    classify_pixels(feature_columns)



























