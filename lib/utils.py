import numpy as np
import os.path as osp
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
# from keras.models import load_model
import keras.optimizers
from keras.engine.network import Network
# noinspection PyPep8Naming
from keras import backend as K
from glob import glob
from collections import defaultdict
import resampy
import wav_manip as wm

# Supported window functions
WINFUNC = {None: None,
           'hamming': np.hamming,
           'hanning': np.hanning,
           'blackman': np.blackman,
           'bartlett': np.bartlett}


# Convert probabilistic prediction to classes
def proba2classes(yproba):
    ycls = np.zeros(yproba.shape, dtype='int8')
    ycls[yproba > 0.5] = 1
    return ycls


# Load data
def load_data(uttlist_fname, spc_dir, peak_dir, tgt_dir, frame_length, winfunc=None):
    # Check window function (None means rectangular window)
    try:
        winfunc = WINFUNC[winfunc]
    except KeyError:
        raise KeyError('Window fuction {} is not supported'.format(winfunc))
    # Read list of utterances to load
    with open(uttlist_fname, 'rt') as fr:
        uttlist = fr.read().splitlines()
    # noinspection PyPep8Naming
    X, y, _ = data_matrices_from_uttlist(uttlist, spc_dir, peak_dir, tgt_dir, frame_length, winfunc=winfunc)
    # noinspection PyPep8Naming
    X = data_as_timesteps(X)
    shape = X.shape[1], X.shape[2]
    return X, y, shape


# Sample-based frame-length features
def data_matrices_from_uttlist(utts, spc_dir, peak_dir, tgt_dir, frame_length, winfunc=None):
    data_list, tgt_list = [], []
    sf = None
    for un in utts:
        # noinspection PyPep8Naming
        # Read samples from an utterance
        spc, sf = wm.read(osp.join(spc_dir, un + '.wav'), 'float')
        fsize = int(frame_length * sf)
        data = wm.frame(spc, np.load(osp.join(peak_dir, un + '.npy')), fsize, winfunc)
        # Collect data
        data_list.append(data)
        tgt_list.append(np.load(osp.join(tgt_dir, un + '.npy')))
    # Make matrices
    y = np.hstack(tgt_list)
    # noinspection PyPep8Naming
    X_data = np.vstack(data_list)
    return X_data, y, sf


def frames_from_utt(samples, sf, frame_pos, frame_length=0.030, winfunc=None):
    fsize = int(frame_length * sf)
    # Append frames from the given utterance
    data_list = wm.frame(samples, frame_pos, fsize, winfunc)
    return data_list


# noinspection PyPep8Naming
def data_as_timesteps(X):
    """Represent data as "timesteps" --> it has the shape (n_samples, timesteps, 1)

    Args:
        X ():

    Returns:
    """
    if isinstance(X, list):
        # Make tabular data from a list of examples
        X = np.vstack(X)
    # Stack data so that features are the 3rd dimension
    return np.dstack([X])


def preprocess(origsamples, sf_src, sf_tgt, filtcoef, norm_amp_spc=30000, norm_amp_filt=0.9):
    samples = downsample(origsamples, sf_src, sf_tgt, norm_amp_spc)
    peaks, filtsamples = detect_peaks(samples, filtcoef, norm_amp_filt)
    return samples, peaks, filtsamples


def detect_peaks(samples, filtcoef, norm_amp_filt=0.9):
    # --- Filter speech and detect peaks
    # Filter speech signal
    filtsamples = wm.filt(samples, filtcoef, zmeansource=True, norm=norm_amp_filt)
    # Convert to int16
    # filtsamples = np.rint(filtsamples).astype('int16')
    filtsamples = wm.float2int(filtsamples)
    # Detect (negative) peaks in filtered signal
    peaks = wm.detect_peaks(filtsamples, peak_polarity='-')
    return peaks, filtsamples


def downsample(origsamples, sf_src, sf_tgt, norm_amp_spc=30000):
    # Downsample signal
    assert sf_src >= sf_tgt, 'Sampling frequency for filtering ({}) is higher than source SF ({}) Hz'. \
        format(sf_tgt, sf_src)
    if sf_tgt != sf_src:
        samples = resampy.resample(origsamples, sf_src, sf_tgt)
    else:
        samples = origsamples
    # Normalize speech signal
    samples = wm.normalize(samples, norm_amp_spc)
    return samples
