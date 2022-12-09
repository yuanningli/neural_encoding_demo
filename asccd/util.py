import os

from functools import wraps
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
from scipy.stats import zscore
import librosa

def save(path, name, *results_keys):
    def decorator(some_function):
        @wraps(some_function)
        def wrapper(*args, **kwargs):
            results = some_function(*args, **kwargs)
            if type(results) is np.ndarray:
                results_dict = {results_keys[0]: results}
            else:
                results_dict = dict(zip(results_keys, results))
            subject = kwargs.pop("subject")
            block = kwargs.pop("block", None)
            subject_block = subject + "_B" + str(block)

            metadata = subject_block
            hz = kwargs.pop("hz", None)
            if hz is not None:
                metadata = metadata + "_" + str(hz) + "hz"

            full_path = os.path.join(path, metadata + "_" + name)
            sio.savemat(full_path, results_dict)
            return results
        return wrapper
    return decorator

def get_subject_block(subject, block):
    return subject + "_B" + str(block)

def get_mel_spectrogram_for_wavpath(wavpath, time_bin=10, n_mels=128):
    fs, y = wavfile.read(wavpath)
    if len(y.shape) > 1 and y.shape[1] == 2:
        y = y[:, 0]
    assert fs/100 == 160
    hz = 1000/time_bin
    assert hz == int(hz)
    hop_length = fs/hz
    assert hop_length == int(hop_length)
    S = librosa.feature.melspectrogram(y=y.astype(np.float), sr=fs, fmax=8000, hop_length=int(hop_length), n_mels=n_mels)
    S = zscore(librosa.power_to_db(S), axis=1)
    return S

def get_mels(n_mels=128, fmin=0, fmax=8000, round=True):
    """Returns center frequencies of mel bands in kHz
    """
    if round:
        return np.around(librosa.core.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax), -2)/1000
    else:
        return librosa.core.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)/1000

def time_to_index(t, hz=100):
    return np.round(t * hz).astype(np.int)

def index_to_time(i, hz=100):
    return i / hz
