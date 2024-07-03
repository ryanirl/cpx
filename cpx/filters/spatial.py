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

from scipy.ndimage import gaussian_filter as _gaussian
from scipy.ndimage import uniform_filter as _uniform
from scipy.ndimage import median_filter as _median

from .parallel import pmap
from ._btype import _btype_filter

__all__ = [
    "gaussian_filter",
    "uniform_filter",
    "median_filter",
]


def _gaussian_filter(
    image: np.ndarray,
    sigma: ArrayLike = 1, 
    *,
    btype: str = "lowpass", 
    mode: str = "nearest", 
    cval: float = 0.0
) -> np.ndarray:
    """Apply a Gaussian filter to an n-dimensional image. 

    Args:
        image (np.ndarray): The input image.
        sigma (Union[float, Sequence[float, float]]): The standard deviation of
            the Gaussian kernel(s).
        btype (str): The type of filter to be applied. Options are 'lowpass',
            'highpass', or 'bandpass'. Default is 'lowpass'.

    Returns:
        np.ndarray: The filtered image.

    """
    sigma = np.atleast_1d(sigma)
    if np.any(sigma <= 0):
        raise ValueError("All values of 'sigma' must be greater than 0.")

    image = image.astype(np.float32)
    image = _btype_filter(
        filter = _gaussian,
        signal = image,
        size = sigma,
        btype = btype,
        target = "sigma",
        mode = mode,
        cval = cval
    )
    return image


def _median_filter(
    image: np.ndarray, 
    size: ArrayLike = 2, 
    *,
    btype: str = "lowpass",
    mode: str = "nearest",
    cval: float = 0.0
) -> np.ndarray:
    """Apply a median filter to the input image using a disk kernel.

    Args:
        image (np.ndarray): The input image.
        size (Union[float, Sequence[float, float]]): The radius of the kernel
            (disk) to take the median over.
        btype (str): The type of filter to be applied. Options are 'lowpass',
            'highpass', or 'bandpass'. Default is 'lowpass'.

    Returns:
        np.ndarray: The filtered image.

    """
    size = np.asarray(size)
    if np.any(size <= 0):
        raise ValueError("All values of 'size' must be greater than 0.")

    image = image.astype(np.float32)
    signal = _btype_filter(
        filter = _median,
        signal = image,
        size = size,
        btype = btype,
        mode = mode,
        cval = cval
    )
    return signal


def _uniform_filter(
    image: np.ndarray,
    size: ArrayLike = 3, 
    *,
    btype: str = "lowpass", 
    mode: str = "nearest", 
    cval: float = 0.0
) -> np.ndarray:
    """Apply a uniform filter to an n-dimensional image. 

    Args:
        image (np.ndarray): The input image.
        size (Union[float, Sequence[float, float]]): The diameter of the uniform
            kernel.
        btype (str): The type of filter to be applied. Options are 'lowpass',
            'highpass', or 'bandpass'. Default is 'lowpass'.

    Returns:
        np.ndarray: The filtered image.

    """
    size = np.asarray(size)
    if np.any(size <= 0):
        raise ValueError("All values of 'size' must be greater than 0.")

    image = image.astype(np.float32)
    image = _btype_filter(
        filter = _uniform,
        signal = image,
        size = size,
        btype = btype,
        mode = mode,
        cval = cval
    )
    return image


# Here we map the spatial filters over the temporal dimension.
gaussian_filter = pmap(_gaussian_filter, axis = 0)
uniform_filter = pmap(_uniform_filter, axis = 0)
median_filter = pmap(_median_filter, axis = 0)


