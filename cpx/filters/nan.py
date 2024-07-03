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


def remove_nans(frames: np.ndarray) -> np.ndarray:
    """Given a 3D input of shape (time, height, width) this function will zero
    all values along the temporal dimension if any NaN values exist aong that
    axis.  This can be a crutial step before filtering because temporal filters
    do not often handle NaN values properly.

    Example:
        >>> array = ... # Shape (time, height, width)
        >>> array = remove_nans(array)

    Args:
        frames (np.ndarray): A numpy array of shape (time, height, width).

    Returns:
        np.ndarray: An NaN filtered numpy array of shape (time, height, width).

    """
    # Integer type NumPy arrays cannot contain NaN values.
    if np.issubdtype(frames.dtype, np.integer):
        return frames

    # Memory-efficient trick to finding NaNs. 
    frames[:, np.isnan(np.max(frames, axis = 0))] = 0
    return frames


