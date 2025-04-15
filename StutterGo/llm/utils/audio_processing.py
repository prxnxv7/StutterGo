import librosa
from scipy.signal import butter, lfilter
import noisereduce as nr

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    return lfilter(b, a, data)

def preprocess_audio(y, sr, target_sr=16000):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    y_filtered = butter_lowpass_filter(y_denoised, cutoff=4000, fs=sr)
    return y_filtered, sr
