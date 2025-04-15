import librosa
import numpy as np

def extract_features_from_segment(y, sr, num_mfcc=40):
    if len(y) < sr * 0.1:
        return None
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    try:
        pitch = np.mean(librosa.yin(y, fmin=50, fmax=500))
    except:
        pitch = 0.0
    return np.concatenate(([zcr, spectral_centroid, pitch], mfcc_mean))
