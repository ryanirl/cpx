# Copyright (c) 2024 Ryan "RyanIRL" Peters
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

from numpy.typing import ArrayLike

from scipy.ndimage import rank_filter as _rank
from scipy.ndimage import median_filter as _median
from scipy.ndimage import uniform_filter1d as _uniform
from scipy.ndimage import gaussian_filter1d as _gaussian
from scipy.signal import savgol_filter as _savgol

from scipy.ndimage import zoom

from ._btype import _btype_filter
from .parallel import pmap
from .utils import bin_array

__all__ = [
    "median_filter",
    "savgol_filter",
    "uniform_filter",
    #"gaussian_filter",
    "rank_filter",
    "baseline_filter"
]


def _median_filter(
    signal: np.ndarray, 
    size: ArrayLike = 3,
    *,
    btype: str = "lowpass",
    mode: str = "nearest",
    cval: float = 0.0,
    axis: int = 0
) -> np.ndarray:
    """

    """
    size = np.atleast_1d(size)
    if np.any(size <= 0):
        raise ValueError("All values of 'size' must be greater than 0.")

    if axis == 0:
        size = [(s, 1) for s in size]
    elif axis == 1:
        size = [(1, s) for s in size]
    else:
        raise ValueError("'axis' must be either 0 or 1.")

    signal = signal.astype(np.float32)
    signal = _btype_filter(
        filter = _median,
        signal = signal,
        size = size,
        btype = btype,
        mode = mode,
        cval = cval
    )
    return signal


def _savgol_filter(
    signal: np.ndarray, 
    size: ArrayLike = 12,
    polyorder: int = 2, 
    *,
    btype: str = "lowpass",
    mode: str = "nearest",
    cval: float = 0.0,
    axis: int = 0
) -> np.ndarray:
    """
    """
    size = np.atleast_1d(size)
    if np.any(size <= 0):
        raise ValueError("All values of 'size' must be greater than 0.")

    signal = signal.astype(np.float32)
    signal = _btype_filter(
        filter = _savgol,
        signal = signal, 
        size = size,
        btype = btype,
        target = "window_length",
        polyorder = polyorder, 
        mode = mode,
        cval = cval,
        axis = axis
    )
    return signal


def _uniform_filter(
    signal: np.ndarray, 
    size: ArrayLike = 9,
    *,
    btype = "lowpass",
    mode = "nearest", 
    cval = 0,
    axis = 0
) -> np.ndarray:
    """ """
    size = np.atleast_1d(size)
    if np.any(size <= 0):
        raise ValueError("All values of 'size' must be greater than 0.")

    signal = signal.astype(np.float32)
    signal = _btype_filter(
        filter = _uniform,
        signal = signal,
        size = size,
        btype = btype,
        mode = mode, 
        cval = cval, 
        axis = axis
    )
    return signal


def _gaussian_filter(
    signal: np.ndarray, 
    sigma: ArrayLike = 9,
    *,
    btype = "lowpass",
    mode = "nearest", 
    cval = 0.0,
    axis = 0
) -> np.ndarray:
    """ """
    sigma = np.atleast_1d(sigma)
    if np.any(sigma <= 0):
        raise ValueError("All values of 'size' must be greater than 0.")

    signal = signal.astype(np.float32)
    signal = _btype_filter(
        filter = _gaussian,
        signal = signal,
        size = sigma,
        btype = btype,
        target = "sigma",
        mode = mode, 
        cval = cval, 
        axis = axis
    )
    return signal


def _rank_filter(
    signal: np.ndarray, 
    size: ArrayLike = 450,
    rank: int = 25,
    *,
    btype: str = "highpass",
    mode: str = "nearest",
    cval: float = 0.0,
    axis: int = 0
) -> np.ndarray:
    """

    """
    size = np.atleast_1d(size)
    if np.any(size <= 0):
        raise ValueError("All values of 'size' must be greater than 0.")

    if axis == 0:
        size = [(s, 1) for s in size]
    elif axis == 1:
        size = [(1, s) for s in size]
    else:
        raise ValueError("'axis' must be either 0 or 1.")

    signal = signal.astype(np.float32)
    signal = _btype_filter(
        filter = _rank,
        signal = signal,
        size = size,
        btype = btype,
        rank = rank,
        mode = mode,
        cval = cval,
    )
    return signal


def _estimate_baseline(
    signal: np.ndarray, 
    size: int = 450,
    percentile: int = 25,
    *,
    mode: str = "edge",
    axis: int = 0,
    **padding_kwargs
) -> np.ndarray:
    """

    """
    ndim = signal.ndim
    signal = signal.astype(np.float32)
    signal, padding = bin_array(
        signal, 
        bin_size = size, 
        axis = axis, 
        mode = mode, 
        return_padding = True, 
        **padding_kwargs
    )
    pad_l, pad_r = padding[axis]
    pad_r = -pad_r if (pad_r != 0) else None

    unpad_idx_ = [slice(None)] * ndim
    unpad_idx_[axis] = slice(pad_l, pad_r)
    unpad_idx = tuple(unpad_idx_)

    axes_ = [1] * ndim
    axes_[axis] = size
    axes = tuple(axes_)

    baseline = np.percentile(signal, percentile, axis = axis + 1)
    baseline = zoom(baseline, axes, mode = "nearest")
    baseline = baseline[unpad_idx]

    return baseline


def _baseline_filter(signal: np.ndarray, *args, **kwargs) -> np.ndarray:
    return signal - _estimate_baseline(signal, *args, **kwargs)


# We map the filters over the height axis. 
median_filter = pmap(_median_filter, axis = 1)
savgol_filter = pmap(_savgol_filter, axis = 1)
uniform_filter = pmap(_uniform_filter, axis = 1)
rank_filter = pmap(_rank_filter, axis = 1)
baseline_filter = pmap(_baseline_filter, axis = 1)


