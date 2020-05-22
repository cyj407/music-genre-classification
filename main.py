import numpy as np
import pandas as pd
# from scipy.io import wavfile
import enum
from enum import Enum
import os
import librosa
import features, saveDataset

class Genre(Enum):
    BLUES = 1
    CLASSICAL = 2
    COUNTRY = 3
    DISCO = 4
    HIPHOP = 5
    JAZZ = 6
    METAL = 7
    POP = 8
    REGGAE = 9
    ROCK = 10

genre_map = {
    'blues': Genre.BLUES,
    'classical': Genre.CLASSICAL,
    'country': Genre.COUNTRY,
    'disco': Genre.DISCO,
    'hiphop': Genre.HIPHOP,
    'jazz': Genre.JAZZ,
    'metal': Genre.METAL,
    'pop': Genre.POP,
    'reggae': Genre.REGGAE,
    'rock': Genre.ROCK,
}

# 5-fold cross validation
'''
method 1
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.8, random_state=888)
'''

# method 2
from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=888)

# for train, test in kfold.split(x, y):

    ### model

    # scores = model.evaluate(x[test], y[test])
    # print(scores)


# training data 40 section in every categories
# test data 10 section in every categories

def getData():

    if(os.path.exists('df_data_no_index.csv')):
        df = pd.read_csv('df_data_no_index.csv')
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
    

def main():
    # print(type(Genre['JAZZ'].value)) # 6 (int)
    df_x, df_y = getData()

    print("5 cross validation")
    # separate to training set and test set
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=888)

    for train_idx, test_idx in kfold.split(df_x, df_y):
        x_train, x_test = df_x.iloc[train_idx], df_x.iloc[test_idx]
        y_train, y_test = df_y.iloc[train_idx], df_y.iloc[test_idx]

        # GMM 

        # scores = model.evaluate(x[test], y[test])
        # print(scores)


if __name__ == '__main__':
    main()