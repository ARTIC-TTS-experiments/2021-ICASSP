# -*- coding: utf-8 -*-
import numpy as np
import logging
from scipy.io import wavfile as wf
from scipy.signal import filtfilt, firls
from itertools import groupby
import librosa as lr
import resampy

"""
Module for speech waveform manipulation.

Attributes:
    logger (:obj:`logging.Logger`): Logger object

"""

logger = logging.getLogger('wav_manip')


def read(fname, dtype='int'):
    """Read speech samples from a filename or from an open file object.

    :obj:`scipy.io.wavfile` or :obj:`librosa.load` is used.

    Args:
        fname (str): file name or open file object
        dtype (str): 'int' for reading using scipy wavfile or 'float' for librosa

    Returns:
        data: data read from waveform
        int: Sampling frequency of waveform
    """
    if dtype == 'float':
        s, sf = lr.load(fname, sr=None)
    elif dtype == 'int':
        sf, s = wf.read(fname)
    else:
        raise TypeError("Supported sample data type is 'float' or 'int' but '{}' was given".format(dtype))
    return s, sf


def resample(x, src_sf, tgt_sf, axis=-1, filter='kaiser_best', **kwargs):
    """Resample speech waveform.

    Args:
        x (:obj:`numpy.array`): The 1-D data to be filtered.
        src_sf (int): Source sampling frequency
        tgt_sf (int):  Target sampling frequency
        axis (int): The target axis along which to resample `x`
        filter (str or callable): The resampling filter to use. By default, uses the `kaiser_best`
                                  (pre-computed filter).
        **kwargs (): Additional keyword arguments provided to the specified filter

    Notes:
        `resampy.resample` is used (https://github.com/bmcfee/resampy)

    Returns:
        :obj:`numpy.array`: The resampled output signal with the same shape as `x`.
    """
    return resampy.resample(x, src_sf, tgt_sf, axis, filter, **kwargs)


def filt(x, numerator, zmeansource=False, norm=0.9):
    """Filter speech waveform to obtain a signal suitable for GCI detection.

    Speech waveform is filtered using a forward-backward filter in a MATLAB-style way and its amplitued normalized
    according to the norm coefficient. Optionally, the filtered signal is zero meaned.

    Args:
        x (:obj:`numpy.array`): The 1-D data to be filtered.
        numerator (:obj:`numpy.array`): Filter numerator coefficients.
        zmeansource (bool): True if to zero mean of the filtered signal
        norm (float): Waveform normalization coefficient.

    Returns:
        :obj:`numpy.array`: The filtered output with the same shape as `x`.

    """
    # Low-pass zero-phase filtering
    y = filtfilt(numerator, 1, x)
    if zmeansource:
        # Zero mean of the filtered signal
        y = zero_mean(y)
    # Return Normalized filtered wav
    return normalize(y, norm)

    # Normalize filtered wav
    # y = norm * np.iinfo(np.int16).max * (y - y.mean()) / max(abs((y - y.mean())))
    # Convert to int16
    # return np.rint(y).astype('int16')


def normalize(x, coef=30000):
    """Simple peak amplitude normalization.

    Args:
        x (:obj:`numpy.array`): The 1-D data.
        coef (float, int): Amplitude coefficient: if float then maximum amplitude is `coef*full_scale`,
                           otherwise maximum is equal to coef.

    Returns:
        :obj:`numpy.array`: The normalized output with the same shape as `x`, converted to int16

    """
    converted = False
    # First: convert to float
    if x.dtype == 'int16':
        x = int2float(x)
        converted = True
    # Convert coef to float
    if isinstance(coef, int):
        coef /= abs(np.iinfo(np.int16).min)
    # Make normalization
    x = x/max(abs(x)) * coef
    # Last: convert back to int if input was int
    if converted:
        x = float2int(x)
    return x


def zero_mean(x):
    """Zero mean of the input signal

    Args:
        x (:obj:`numpy.array`): The 1-D data to be filtered.

    Returns:
        :obj:`numpy.array`: The zero mean output with the same shape as `x`.

    """
    return (x - x.mean()) / max(abs((x - x.mean())))


def design_highpass(sf, order=10001, stop_freq=70, pass_freq=100):
    """Design high-pass filter.

    Args:
        sf (int):        Sampling frequency.
        order (int):     The number of taps (order) of the filter.
        stop_freq (int): The frequency below which must the filter MUST act like as stop filter.
        pass_freq (int): The frequency above which the filter MUST act like a pass filter.

    Returns:
        :obj:`numpy.array`: Array of numerator coefficients

    """
    nyquist_rate = sf/2.
    desired = (0, 0, 1, 1)
    bands = (0, stop_freq, pass_freq, nyquist_rate)
    return firls(order, bands, desired, nyq=nyquist_rate)


def detect_peaks(x, peak_polarity='-'):
    """Detect both negative and positive peaks in the input speech waveform.

    The negative polarity is marked as '-', 'n', 'neg', or 'min.
    The positive polarity is marked as '+', 'p', 'pos', or 'max.

    Args:
        x (:obj:`numpy.array`): Input speech signal.
        peak_polarity (str): Polarity of the peaks to detect (negative/positive)

    eturns:
        :obj:`numpy.array`: Array of indeces of negative/positive peaks in the signal.

    """
    # Zero-cross the filtered signal
    zc = zerocross(x)

    if peak_polarity in ('-', 'n', 'neg', 'min'):
        return _detect_negpeaks(x, zc)
    elif peak_polarity in ('+', 'p', 'pos', 'max'):
        return _detect_pospeaks(x, zc)
    else:
        raise RuntimeError('Unknown type of signal polarity')


def zerocross(x):
    """Zero-cross the input signal.

    Args:
        x (:obj:`numpy.array`): Input speech signal.

    Returns:
        :obj:`list`: List of indeces of elements after which a zero crossing occurs.

    """
    # Make signum in the input segment
    seg = np.sign(x)

    # Replace zeros with -1 to correctly interpret zeros in the input segment
    seg[seg == 0] = -1

    # Detect zerocross as a list of indeces of elements after which a zero crossing occurs
    return np.where(np.diff(seg))[0]


def _detect_negpeaks(x, zc):
    """Detect negative peaks.

    Args:
        x (:obj:`numpy.array`): Input speech signal.
        zc (:obj:`list`): List of indeces of elements after which a zero crossing occurs.

    Returns:
        :obj:`numpy.array`: Array of indeces of negative peaks in the signal

    """
    peaks = []
    # Go through all zerocrossing indeces
    for i_z, i_s in enumerate(zc[:-1]):
        # Setup indeces
        c = i_s  # index of the current zerocross in signal
        n = zc[i_z + 1]  # index of the next zerocross in signal
        # find the peak between two zerocrosses
        if x[c] > 0:
            # for signal under zero, the current zerocrossing was the last with a positive value
            # find the highest negative peak
            arg = np.argmin(x[c:n])
            # find the length of the sequence of the same highest peaks if any
            seqlen = len([list(j) for i, j in groupby(x[c + arg:n])][0])
            # pick up the middle index of the same peaks
            mid = arg + seqlen // 2
            # store negative peak
            peaks.append(c + mid)
    return np.array(peaks)


def _detect_pospeaks(x, zc):
    """Detect positive peaks.

    Args:
        x (:obj:`numpy.array`): Input speech signal.
        zc (:obj:`list`): List of indeces of elements after which a zero crossing occurs

    Returns:
        :obj:`numpy.array`: Array of indeces of positive peaks in the signal

    """
    peaks = []
    # Go through all zerocrossing indeces
    for i_z, i_s in enumerate(zc[:-1]):
        # Setup indeces
        c = i_s  # index of the current zerocross in signal
        n = zc[i_z + 1]  # index of the next zerocross in signal
        # find the peak between two zerocrosses
        if x[c] <= 0:
            # for signal above zero, the current zerocrossing was the last with a non-positive value
            # find the highest positive peak
            arg = np.argmax(x[c:n])
            # find the length of the sequence of the same highest peaks if any
            seqlen = len([list(j) for i, j in groupby(x[c + arg:n])][0])
            # pick up the middle index of the same peaks
            mid = arg + seqlen // 2
            # store positive peak
            peaks.append(c + mid)
    return np.array(peaks)


def frame(x, peaks, fsize, winfunc=None):
    """Divide speech signal into possibly overlapping frames.

    Args:
        x (:obj:`numpy.array`): Input speech signal.
        peaks (:obj:`numpy.array`:): Array of indeces of negative/positive peaks in the signal.
        fsize (int): Frame size in No. of samples
        winfunc (func): Window function or None

    Returns:
        :obj:`numpy.array`: Matrix of peak-based frames

    """
    # Half of the windows size
    n2 = fsize//2
    # Signal size
    ssize = len(x)
    # Init output array
    # noinspection PyPep8Naming
    X = np.zeros((len(peaks), fsize+1), dtype=x.dtype)
    # Peak-by-peak framing
    for idx, p in enumerate(peaks):
        # Set start sample at the beginning
        (b, ib) = (0, n2-p) if p-n2 < 0 else (p-n2, 0)
        # Set end sample at the end
        (e, ie) = (ssize, fsize+1-(p+n2+1-ssize)) if p+n2 >= ssize else (p+n2+1, fsize+1)
        # Extract samples around the given peak
        X[idx][ib:ie] = x[b:e]
    # :-1 is to secure even number of samples especially for FFT computation
    return X[:, :-1] if winfunc is None else np.multiply(X[:, :-1], winfunc(fsize))


# noinspection PyPep8Naming
def melspec(X, sf, n_fft=512, power=2.0, n_mels=128, to_db=True):
    """Compute mel spectral amplitudes for each input frame.

    Args:
        X (:obj:`numpy.array`): Frames of speech samples
        sf (int): Sampling frequency
        n_fft (int): length of the FFT window
        power (float): Exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power, etc.
        n_mels (int): Number of Mel bands to generate
        to_db (bool): Whether to convert amplitude to dB

    Returns:
        :obj:`numpy.array` (shape=(X.shape[0], n_mels): Frames of mel spectrograms

    """
    S = np.apply_along_axis(lr.feature.melspectrogram, 1, X, sf, n_fft=n_fft, hop_length=n_fft + 1,
                            power=power, n_mels=n_mels)
    S = S.reshape((X.shape[0], n_mels))
    if to_db:
        S = np.apply_along_axis(lr.core.amplitude_to_db, axis=1, arr=S)
        S = S.reshape((X.shape[0], n_mels))
    return S


def int2float(x):
    return x / abs(np.iinfo(np.int16).min)


def float2int(x):
    return np.rint(x * abs(np.iinfo(np.int16).min)).astype('int16')
