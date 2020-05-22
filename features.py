from librosa.onset import onset_strength
from librosa import feature, beat, decompose
import pywt
import librosa
import math
import numpy as np
from scipy.signal import hilbert
import pandas as pd

hop_size = 512

def feature_low_energy(wave):
    rms_per_win = feature.rms(y=wave, hop_length=hop_size)
    # print(len(rms_per_win[0]))   # total analysis windows
    # every 43 analysis window forms a texture window
    num_texture_win = 30
    total_rms_texture_win = np.zeros(num_texture_win)
    k = 0
    for i in range(num_texture_win):
        total_rms_texture_win[k] = np.sum(rms_per_win[0][43 * i: 43 * (i+1)])
    avr_texture_win = np.sum(total_rms_texture_win) / num_texture_win
    
    # analysis windows rms energy < average of texture window
    p = sum(analysis_win < avr_texture_win for analysis_win in rms_per_win[0]) / len(rms_per_win[0])
    return p

def findTimbral(wave):  # 19 dimensions
    print('Finding Timbral Features....')
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

    five_mfcc = feature.mfcc(wave, n_mfcc=5)    # 10 dim
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
    # rhythm_feature.append(np.sum(tempogram))

    # rhythm_feature['tempo_max'] = np.max(tempogram)
    # rhythm_feature['histo_peak_bin'] = tempogram[1]
    # tempo_freq = librosa.tempo_frequencies(tempogram.shape[0])
    # rhythm_feature['tempo_freq'] = np.mean(tempo_freq[1:]) # 44.032683 
    # rhythm_feature.append(np.max(tempo_freq))

    # global
    # autocorr = librosa.autocorrelate(env, max_size=tempogram.shape[0])
    # autocorr = librosa.util.normalize(autocorr)
    # global_tempo = librosa.beat.tempo(env, hop_length=hop_size)[0]  #  139.60114

    tempo, beats = librosa.beat.beat_track(wave, hop_length=hop_size)
    rhythm_feature['tempo'] = tempo
    # rhythm_feature.append(tempo)

    timestamp = librosa.frames_to_time(beats, hop_length=hop_size)
    interval = []
    for i in range(1, len(timestamp)):
        interval.append(timestamp[i] - timestamp[i-1])
    period = sum(interval) / len(interval)
    rhythm_feature['tempo_period'] = period
    # rhythm_feature.append(period)
    return rhythm_feature

# def findPitch(wave):
#     pitch_feature = []
#     pitches, magnitudes = librosa.piptrack(wave, n_fft=512, fmin=27, fmax=4186)
#     n = librosa.hz_to_midi(pitches)
#     c = np.mod( np.multiply((np.mod(n, 12) , 7), 12))
#     modified_fph = np.zeros(12)
#     for i in range(len(magnitudes)):
#         modified_fph[c[i]] = modified_fph[c[i]] + magnitudes[i]
    
#     pitch_feature.append(np.max(modified_fph))  # max amplitude in the histogram
#     pitch_feature.append(np.sum(modified_fph))  # sum of the histogram

def extract(wave):
    print('feature extraction')
    feature = {}

    timbral = findTimbral(wave)
    feature.update(timbral)

    rhythm = findRhythmic(wave)
    feature.update(rhythm)
    
    # findPitch(wave)
    return feature