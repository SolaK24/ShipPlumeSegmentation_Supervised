import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.model_selection import RandomizedSearchCV,  GroupKFold
from model_params import model_dict, feature_columns

np.random.seed(1234)



def feature_importance_plot(d, feature_columns):
    feature_columns_nice = ["Moran's I", "NO2", 'Level_0', 'Level_1', 'Level_2', 'Level_3', 'Level_4', 'Level_5',
                            'Sector_0', 'Sector_1', 'Sector_2', 'Sector_3',
                            'Wind_spd', 'Wind_dir_sin', 'Wind_dir_cos', 'Ship_length', 'Ship_spd_avg']

    fig, axs = plt.subplots(1, 4, figsize=(16, 4), facecolor='w', edgecolor='w')
    fig.subplots_adjust(hspace=.0001)
    i = 0
    df_feature_importance = pd.DataFrame()
    for model_name in list(d.keys()):

        if not (model_name.startswith('scaler') or model_name.startswith('SVM')):
            df = pd.DataFrame()
            if model_name in ['Linear_SVM', 'Logistic']:
                axs[i].barh(feature_columns, d[model_name]['model'].coef_[0], alpha=0.5)
                axs[i].set_title(f'{model_name}', fontsize=14)
                axs[i].set_xlim(-0.2, 0.55)
                axs[i].set_xlabel('Model coeficients', fontsize=14)
                axs[i].xaxis.set_label_coords(0.5, -0.09)
                df['Coefs'] = d[model_name]['model'].coef_[0]
                df['Feature_names'] = feature_columns
                df['Model'] = model_name


            else:
                axs[i].barh(feature_columns, d[model_name]['model'].feature_importances_,
                            alpha=0.5, log=True)
                axs[i].set_title(f'{model_name}', fontsize=14)
                axs[i].set_xlabel('Feature importance - log scale', fontsize=14)
                axs[i].xaxis.set_label_coords(0.52, -0.09)
                df['Coefs'] = d[model_name]['model'].feature_importances_
                df['Feature_names'] = feature_columns
                df['Model'] = model_name


            axs[i].grid(alpha=0.2)

            if i > 0:
                axs[i].set_yticklabels([])
            else:
                axs[i].set_yticklabels(feature_columns_nice, fontsize=14)
            i += 1
            df_feature_importance = df_feature_importance.append(df, ignore_index=True)

    plt.savefig(f'./clf_results/feature_importance.png', bbox_inches='tight')
    plt.close()
    df_feature_importance.to_csv(f'./clf_results/feature_importance_df.csv')

def grid_search(train_set, class_weight,  feature_columns):

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
                                n_iter=60,
                                n_jobs=3, error_score='raise', scoring='average_precision')
        scaler = preprocessing.StandardScaler()
        gs.fit(scaler.fit_transform(train_set[feature_columns].values),
               train_set.Label.values.reshape(-1,1))
        with open(f'./{model_path}/scaler.pkl', 'wb') as fid:
            pickle.dump(scaler, fid)
            fid.close()

        pd.DataFrame(gs.cv_results_).to_csv(f'./clf_results/{model_name}_cv_results.csv')
        best_model_dict[model_name]['model'] = gs.best_estimator_
        best_model_dict[model_name]['param_dict'] = gs.best_params_
        with open(f'./best_models/{model_name}.pkl', 'wb') as fid:
            pickle.dump(best_model_dict[model_name]['model'], fid)
            fid.close()
    feature_importance_plot(best_model_dict, feature_columns)




def classify_pixels(feature_columns):

    labeled_df = pd.read_csv('data/labeled_data.csv')
    df_model = labeled_df[labeled_df.sector == 1].reset_index(drop=True)
    weight_for_0 = (1 / len(df_model[df_model.Label == 0])) * (len(df_model) / 2.0)
    weight_for_1 = (1 / len(df_model[df_model.Label == 1])) * (len(df_model) / 2.0)
    class_weight_dict = {0: weight_for_0, 1: weight_for_1}
    grid_search(df_model, class_weight_dict, feature_columns)



if __name__ == '__main__':
    classify_pixels(feature_columns)