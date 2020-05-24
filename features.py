from librosa.onset import onset_strength
from librosa import feature, beat, decompose
import librosa
import math
import numpy as np
from scipy.signal import hilbert
import pandas as pd
hop_size = 512

def feature_low_energy(wave):
    rms_per_win = feature.rms(y=wave, hop_length=hop_size)
    num_texture_win = 30
    analy_in_tex = 43   # every 43 analysis window forms a texture window
    total_rms_texture_win = np.zeros(num_texture_win)
    for i in range(num_texture_win):
        total_rms_texture_win[i] = np.sum(
            rms_per_win[0][analy_in_tex * i: analy_in_tex * (i+1)]) / analy_in_tex
    avr_texture_win = np.sum(total_rms_texture_win) / num_texture_win

    # analysis windows rms energy < average of texture window
    p = sum(analysis_win < avr_texture_win for analysis_win in rms_per_win[0]) / len(rms_per_win[0])
    return p

def findTimbral(wave):  # 19 dimensions
    timbral_feature = {}

    centroid = feature.spectral_centroid(wave)
    timbral_feature['mu_centroid'] = np.mean(centroid)
    timbral_feature['var_centroid'] = np.var(centroid, ddof=1)

    rolloff = feature.spectral_rolloff(wave)
    timbral_feature['mu_rolloff'] = np.mean(rolloff)
    timbral_feature['var_rolloff'] = np.var(rolloff, ddof=1)
    
    flux = onset_strength(wave, lag=1)    # spectral flux
    timbral_feature['mu_flux'] = np.mean(flux)
    timbral_feature['var_flux'] = np.var(flux, ddof=1)

    zero_crossing = feature.zero_crossing_rate(wave)
    timbral_feature['mu_zcr'] = np.mean(zero_crossing)
    timbral_feature['var_zcr'] = np.var(zero_crossing)

    five_mfcc = feature.mfcc(wave, n_mfcc=10)    # n_mfcc = 10 dim
    i = 1
    for coef in five_mfcc:
        timbral_feature['mu_mfcc' + str(i)] = np.mean(coef)
        timbral_feature['var_mfcc' + str(i)] = np.var(coef, ddof=1)
        i = i + 1

    percent = feature_low_energy(wave)          # 1 dim
    timbral_feature['low_energy'] = percent    

    return timbral_feature

def findRhythmic(wave): # 3 dimensions
    rhythm_feature = {}
    
    env = onset_strength(wave)
    tempogram = feature.tempogram(onset_envelope=env, hop_length=hop_size)
    rhythm_feature['tempo_sum'] = np.sum(tempogram)

    return rhythm_feature

def extract(wave):
    feature = {}

    timbral = findTimbral(wave)
    feature.update(timbral)

    rhythm = findRhythmic(wave)
    feature.update(rhythm)

    return feature