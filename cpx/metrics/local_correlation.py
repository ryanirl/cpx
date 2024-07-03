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

from scipy.ndimage import convolve


def local_correlation_projection(frames, kernel_size = 5, *, verbose = True):
    """Memory and time efficient local correlation algorithm. It makes use of
    the fact that a correlation between to signals is a normalized dot product,
    and that the dot product is an inherently iterative procedure. From the dot
    product perspective, a correlation between one pixel's temporal trace and 
    anothers is just a running sum of their time points multiplied by each other
    after mean and unit-one normalization. Thus a local correlation is just this
    operation applied with every pixel and it's neighbors within some pre-defined 
    radius, followed by a mean at the end. This can further be optimized when you
    realize that a convolution with a 0 center uniform fillter was just described.
    
    The extra space required for this algorithm is roughly `frames[i] * 5` for 
    some arbitrary `i`. Unlike other algorithms, this implementation does not
    scale with the size of `frames`, but instead with the size of each individual 
    frame given that it's iterative.

    In my testing, for a 20 gigabyte video, running this function didn't raise 
    my RAM over extra gigabyte using a kernel size of 5. Furthermore, it was
    faster than any other implementation I have used. 

    Args:
        frames (np.ndarray): A numpy movie of shape (n_frames, height, width).
        kernel_size (int): The size in diameter (not radius) of the nearest 
            neighbor kernel to use. 
        verbose (bool): Whether to show the TQDM loading bars or not.

    Returns:
        np.ndarray: A two-dimensional array representing the local correlation
            projection of `frames`.

    """
    if kernel_size % 2 == 0:
        raise ValueError("'kernel_size' must be odd.")

    n_frames, height, width = frames.shape

    # Compute the kernel for the per-frame convolution. 
    kernel = np.ones((kernel_size, kernel_size), dtype = np.float32)
    kernel[kernel_size // 2, kernel_size // 2] = 0
    kernel = kernel / kernel.sum()

    mean = frames.mean(axis = 0)
    
    # Using np.linalg.norm(frames, axis = 0) is not memory efficient so we
    # vectorize it only over one dimension.
    norm = np.zeros((height, width), dtype = np.float32) 
    for i in tqdm(range(frames.shape[1]), disable = not verbose):
        norm[i] = np.linalg.norm(frames[:, i] - mean[i], axis = 0)

    # Now compute the local correlation as a running sum of normalized dot
    # products of each pixel with it's nearest neighbors.
    local_corr = np.zeros((height, width), dtype = np.float32)
    for i in tqdm(range(n_frames), disable = not verbose):
        image = (frames[i].copy() - mean) / norm
        image = image * convolve(image, kernel, mode = "constant")
        local_corr += image

    return local_corr


def pixel_local_correlation(frames, inds):
    """Returns the correlation of a single pixel across time with the other
    pixels. See the documentation of `local_correlation_projection()` for
    information about the algorithm it uses. 
    
    Args:
        frames (np.ndarray): A numpy movie of shape (n_frames, height, width).
        inds (Tuple[int, int]): A size two tuple representing the indices of the
            pixel to correlate the whole movie with. For example `inds = (5, 10)`
            would correspond to the the pixel `frames[:, 5, 10]`.

    Returns:
        np.ndarray: A two-dimensional array representing the correlation of
            every pixel with the pixel defined by `inds`.
    
    """
    return trace_local_correlation(frames, frames[:, inds[0], inds[1]])


def trace_local_correlation(frames, trace):
    """Returns the correlation of a time series `trace` with every pixel across
    time in `frames`. See the documentation of `local_correlation_projection()`
    for information about the algorithm it uses. 
    
    Args:
        frames (np.ndarray): A numpy movie of shape (n_frames, height, width).
        trace (np.ndarray): A time series, with the same length of the number of
            frames in `frames`, to correlate with every pixel in `frames` across
            time. 

    Returns:
        np.ndarray: A two-dimensional array representing the correlation of
            every pixel with the time series `trace`.
    
    """
    n_frames, height, width = frames.shape
    if len(trace) != n_frames:
        raise ValueError(
            f"The length of input trace ({len(trace)}) does not match the "
            f"number of frames ({n_frames})."
        )
    
    mean = frames.mean(axis = 0)

    # Using np.linalg.norm(frames, axis = 0) is not memory efficient so we
    # vectorize it only over one dimension.
    norm = np.zeros((height, width), dtype = np.float32) 
    for i in tqdm(range(frames.shape[1])):
        norm[i] = np.linalg.norm(frames[:, i] - mean[i], axis = 0)
    
    # Normalize the trace before-hand.
    trace = (trace - trace.mean()) / np.linalg.norm(trace)

    corr = np.zeros((height, width), dtype = np.float32)
    for i in tqdm(range(n_frames)):
        image = (frames[i].copy() - mean) / norm
        image = image * trace[i]
        corr += image
        
    return corr


