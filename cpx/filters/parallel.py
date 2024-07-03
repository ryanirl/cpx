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

from joblib import delayed, Parallel
from functools import wraps
from tqdm.auto import tqdm


def pmap(func, axis = 0):
    @wraps(func)
    def wrapper(frames, *args, n_jobs = 1, **kwargs):
        """This wrapper/decorator parallelizes any function over a single 
        dimension of a video stack. In some cases this can lead to a 14x speed 
        up of non-thread optimized code such as the scipy spatial gaussian and
        median filters. 

        The function arguments are adjusted to include the `n_jobs` argument,
        which by default is set to 1. 

        Example:
        
            >>> # Unoptimized code to parallelize.
            >>> def apply_spatial_func()
            >>>     # Adds 1 to every frame.
            >>>     for i in range(len(frames)):
            >>>         frames[i] = frames[i] + 1 
            >>>     return frames

            >>> # Parallelized code [Option #1]:
            >>> def add_one_to_frame(frame):
            >>>     return frame + 1
            >>>
            >>> apply_spatial_func = pmap(add_one_to_frame, axis = 0)

        """
        if frames.ndim != 3:
            raise ValueError(
                "'frames' must be 3 dimensional. The recommended shape is "
                "(n_frames, height, width)."
            )

        if axis not in [0, 1, 2]:
            raise ValueError(
                "Given that 'frames' is 3D, 'axis' must be 0, 1, or 2."
            )

        def _pcall(i):
            if axis == 0:
                frames[i] = func(frames[i], *args, **kwargs)
            elif axis == 1:
                frames[:, i] = func(frames[:, i], *args, **kwargs)
            else:
                frames[:, :, i] = func(frames[:, :, i], *args, **kwargs)

        # We only use parallel if 'n_jobs' is not 1. 
        if (n_jobs != 1):
            # Because numpy array indexing is performed in `pcall`, it is required
            # to use shared memory. Thus we force the backend to be `threading`.
            with Parallel(n_jobs = n_jobs, require = "sharedmem", backend = "threading") as parallel:
                parallel(delayed(_pcall)(i) for i in tqdm(range(frames.shape[axis])))
        else:
            for i in tqdm(range(frames.shape[axis])):
                _pcall(i)

        return frames
    return wrapper


