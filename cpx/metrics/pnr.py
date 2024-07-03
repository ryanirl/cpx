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

from tqdm.auto import tqdm
import numpy as np

from typing import Tuple

from scipy.signal import welch


def estimate_noise_std(
    signal: np.ndarray, 
    freq_range: Tuple[float, float] = (0.5, 1.0), 
    fs: int = 15, 
    nperseg: int = 1024, 
    axis: int = -1
) -> np.ndarray:
    """Estimate the noise of an n-dimensional signal along an arbitrary axis
    `axis` by averaging the PSD over some range (given by `freq_range`).

    Args:
        signal (np.ndarray): 
        freq_range (Tuple[float, float]): 
        fs (int): 
        nperseg (int): 
        axis (int): 

    Returns:
        Union[float, np.ndarray]

    """
    f, psd = welch(signal, fs = fs, nperseg = nperseg, axis = axis)
    
    idx_l = int(freq_range[0] * len(f))
    idx_r = int(freq_range[1] * len(f))

    # Dynamic slicing of a numpy array across some axis. This is more efficient
    # than np.take(...) because it creates a view rather than copying the data.
    idx_ = [slice(None)] * signal.ndim
    idx_[axis] = slice(idx_l, idx_r)
    idx = tuple(idx_)
    
    psd = psd[idx]
    f = f[idx_l : idx_r]
    
    noise_std = np.trapz(psd, f, axis = axis)
    noise_std = np.sqrt(noise_std)
    return noise_std


def get_pnr(traces: np.ndarray, **kwargs) -> np.ndarray:
    if traces.ndim != 2:
        raise ValueError(
            "The traces must have `ndim == 2` and must be of shape (n, time)."
        )

    # Get the maximum projection and noise estimate for each trace.
    max_proj = np.max(traces, axis = 1)
    std_proj = estimate_noise_std(traces, **kwargs)

    # Peak to noise is defined as `peak / noise`. 
    pnr = np.divide(max_proj, std_proj)
    pnr[pnr < 0] = 0
    return pnr


def pnr_projection(frames: np.ndarray, verbose: bool = True) -> np.ndarray:
    t, h, w = frames.shape
    
    pnr = np.zeros((h, w), dtype = np.float32)
    for i in tqdm(range(h), disable = not verbose):
        pnr[i] = get_pnr(frames[:, i].T).T

    return pnr


