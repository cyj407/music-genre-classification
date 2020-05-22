import numpy as np
import pandas as pd
# from scipy.io import wavfile
import enum
from enum import Enum
import os
import librosa
import features

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

def dataset():
    print("read wav files")
    x = []
    y = []
    # wav_path = []
    path = os.getcwd() + '\\res\\'
    for r, d, f in os.walk(path):
        for i in f:
            if('.wav' in i):
                # wav_path.append(os.path.join(r, i))
                wav_array, sr = librosa.load(os.path.join(r, i), sr=22050)
                # sr, wav_array = wavfile.read(os.path.join(r, i))
                x.append(wav_array) # np array

                genre_name = r.split('\\res\\')[1]
                y.append(genre_map[genre_name].value) # integer
                
    # print(len(wav_path))
    # for i in range(len(x)):
    #     print(x[i])
    #     print(y[i])
    # np_x = np.asarray(x)
    # np_y = np.asarray(y)
    # print(x)
    # print(np_x)
    return x, y


def main():
    # print(type(Genre['JAZZ'].value)) # 6 (int)

    # construct dataset
    signal, y = dataset()

    # construct features
    df = pd.DataFrame()
    for i in range(4, len(signal), 50):        
        new_x = pd.DataFrame(features.extract(signal[i]), index=[i])
        df = df.append(new_x)
    
    print(df.head(10))

    # separate to training set and test set
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=888)

    # for train_idx, test_idx in kfold.split(x, y):

        # x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        # y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # print(y_train)
        # print(y_test)

        # GMM !

        # scores = model.evaluate(x[test], y[test])
        # print(scores)


if __name__ == '__main__':
    main()