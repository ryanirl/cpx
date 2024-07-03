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

from numpy.typing import ArrayLike # Not sure this is the way.
from typing import Callable


def _btype_filter(
    *,
    filter: Callable, 
    signal: np.ndarray, 
    size: ArrayLike,
    btype: str,
    target: str = "size",
    **kwargs
) -> np.ndarray:
    """Helper function to reduce boilerplate. It accepts a filter function,
    btype string, and an iterable filter size, then filters the data `signal`
    with `filter` depending on `btype` and `size`. 

    """
    size = np.atleast_1d(size)

    if btype == "lowpass":
        if len(size) != 1: 
            raise ValueError("Must specify a single size for lowpass filters.")

        kwargs.update({target: size[0]})
        return filter(signal, **kwargs)

    elif btype == "highpass":
        if len(size) != 1: 
            raise ValueError("Must specify a single size for highpass filters.")

        kwargs.update({target: size[0]})
        return signal - filter(signal, **kwargs)

    elif btype == "bandpass":
        if len(size) != 2: 
            raise ValueError("Must specify two sizes for bandpass filters.")

        kwargs_l = kwargs.copy()
        kwargs_h = kwargs.copy()
        kwargs_l.update({target: size[0]})
        kwargs_h.update({target: size[1]})
        return filter(signal, **kwargs_l) - filter(signal, **kwargs_h)

    else:
        raise ValueError(
            "Invalid 'btype'. Supported types are 'lowpass', 'highpass', or 'bandpass'."
        )


