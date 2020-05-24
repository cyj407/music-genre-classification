from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import enum
from enum import Enum
import os
import librosa
import features, saveDataset
from sklearn import metrics, preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df_column = ['mu_centroid', 'var_centroid',
        'mu_rolloff', 'var_rolloff',
        'mu_flux', 'var_flux',
        'mu_zcr', 'var_zcr',
        'mu_mfcc1', 'var_mfcc1',
        'mu_mfcc2', 'var_mfcc2',
        'mu_mfcc3', 'var_mfcc3',
        'mu_mfcc4', 'var_mfcc4',
        'mu_mfcc5', 'var_mfcc5',
        'mu_mfcc6', 'var_mfcc6',
        'mu_mfcc7', 'var_mfcc7',
        'mu_mfcc8', 'var_mfcc8',
        'mu_mfcc9', 'var_mfcc9',
        'mu_mfcc10', 'var_mfcc10',
        'low_energy',
        'tempo', 'tempo_period',
        'tempo_sum'
        ]

select_feature = ['mu_centroid', 'var_centroid',
        'mu_rolloff', 'var_rolloff',
        'mu_flux', 'var_flux',
        'mu_zcr', 'var_zcr',
        'mu_mfcc1', 'var_mfcc1',
        'mu_mfcc2', 'var_mfcc2',
        'mu_mfcc3', 'var_mfcc3',
        'mu_mfcc4', 'var_mfcc4',
        'mu_mfcc5', 'var_mfcc5',
        'mu_mfcc6', 'var_mfcc6',
        'mu_mfcc7', 'var_mfcc7',
        'mu_mfcc8', 'var_mfcc8',
        'mu_mfcc9', 'var_mfcc9',
        'mu_mfcc10', 'var_mfcc10',
        'low_energy',
        # 'tempo', 'tempo_period',      # lower
        'tempo_sum'
        ]

def getData():

    if(os.path.exists('df_data_no_index_10mfcc.csv')):
        df = pd.read_csv('df_data_no_index_10mfcc.csv')
        df_y = df['genre']
        df_x = df.drop('genre', axis=1)
        return df_x, df_y
    
    # features haven't been created
    print("Create Dataset")
    signal, y = saveDataset.dataset()
    df_y = pd.DataFrame(data=y, columns=['genre'])

    # construct features
    print("Feature Extraction")
    df_x = pd.DataFrame()
    for i in range(0, len(signal)):        
        new_x = pd.DataFrame(features.extract(signal[i]), index=[i])
        df_x = df_x.append(new_x)
        
    # saveDataset.saveFeature(df_x, df_y)
    return df_x, df_y
    
def trainModel(df_x, df_y):

    df_x = df_x[select_feature]

    train_acc_list = []
    test_acc_list = []

    # 5 Cross Validation - separate to training set and test set
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=888)
    for train_idx, test_idx in kfold.split(df_x, df_y):
        train_x, test_x = df_x.iloc[train_idx], df_x.iloc[test_idx]
        train_y, test_y = df_y.iloc[train_idx], df_y.iloc[test_idx]

        clf1 = RandomForestClassifier(
            n_estimators=250,
            criterion='entropy',
            min_weight_fraction_leaf=0.01,
            random_state=2000)    # 0.738
        # model = clf1

        clf2 = SVC(C=60.0, kernel='rbf', gamma=0.1, random_state=2000)   # 0.758
        # model = clf2

        clf3 = LogisticRegression(
            penalty='l2',
            C=300.0,
            solver='lbfgs',
            max_iter=1000,
            multi_class='multinomial',
            random_state=2000)
        # model = clf3

        model = VotingClassifier(   # 0.782
            estimators=[('rf', clf1), ('svc', clf2), ('lr', clf3)], voting='hard') 

        model.fit(train_x, train_y)
        train_pred_y = model.predict(train_x)
        train_acc = metrics.accuracy_score(train_y, train_pred_y)
        test_pred_y = model.predict(test_x)
        test_acc = metrics.accuracy_score(test_y, test_pred_y)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    print(train_acc_list)
    print(test_acc_list)
    print('average train accuracy: {}\naverage validate accuracy: {}'.format(
        np.mean(train_acc_list), np.mean(test_acc_list)
    ))

    return model

def plotImportance(model):
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=select_feature)
    feat_importances.nlargest(len(select_feature)).plot(kind='barh')
    plt.show()


def main():
    df_x, df_y = getData()

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    dataset_numpy = scaler.fit_transform(df_x.to_numpy())
    df_x = pd.DataFrame(dataset_numpy, columns=df_column)

    model = trainModel(df_x, df_y)

if __name__ == '__main__':
    main()