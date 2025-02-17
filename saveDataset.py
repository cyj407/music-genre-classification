import numpy as np
import pandas as pd
import enum
from enum import Enum
import os
import librosa
import features
import sklearn.preprocessing

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


def dataset():
    print("read wav files")
    x = []
    y = []
    path = os.getcwd() + '\\res\\'
    for r, d, f in os.walk(path):
        for i in f:
            if('.wav' in i):
                wav_array, sr = librosa.load(os.path.join(r, i), sr=22050)
                x.append(wav_array) # np array

                genre_name = r.split('\\res\\')[1]
                y.append(genre_map[genre_name].value) # integer
                
    return x, y

def saveFeature(x, y):
    x['genre'] = y    
    x.to_csv(os.getcwd() + '\\df_no_index_10mfcc.csv', index=0)

def main():
    print(os.getcwd())
    print("Create Dataset")
    signal, y = dataset()
    df_y = pd.DataFrame(data=y, columns=['genre'])

    # construct features
    print("Feature Extraction")
    df_x = pd.DataFrame()
    for i in range(0, len(signal)):
        print("number "+ str(i))
        new_x = pd.DataFrame(features.extract(signal[i]), index=[i])
        df_x = df_x.append(new_x)

    saveFeature(df_x, df_y)

if __name__ == '__main__':
    main()