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
import logging

from joblib import Parallel
from joblib import delayed

from scipy.ndimage import gaussian_filter

from .rigid import get_rigid_shift
from .rigid import apply_rigid_shift

logger = logging.getLogger(__name__)


def rigid_motion_correction(frames, template = None, return_shifts = False, **kwargs):
    """A functional definition of rigid motion correction."""
    model = RigidMotionCorrection(**kwargs)
    frames = model.fit_transform(frames, template)

    if return_shifts:
        return frames, np.array(model.rigid_shifts)

    return frames


def gaussian_bandpass_filter(image, sigma):
    """Spatial bandpass filter by a difference of gaussians.

    The purpose of copying this here is to reduce interdependencies on other cpx
    packages such as `cpx.filters`. 
    
    """
    return gaussian_filter(image, sigma[0]) - gaussian_filter(image, sigma[1])


class RigidMotionCorrection:
    def __init__(
        self,
        max_shifts = (15, 15), 
        upsample_factor_fft = 10,
        template_range = (0, 1),
        spatial_bandpass = True,
        sigma = (0.5, 8.0),
        add_min = True,
        parallel_backend = "threading",
        n_jobs = 1,
        verbose = True
    ):
        """A scipy-like model rigid motion correction model with `fit()`,
        `transform()`, and `fit_transform()` methods. 

        Args:
            max_shifts (Tuple[int, int]): The maximum shift allowed, it is
                recommended to set this as to not get exploding and unrealistic
                shifts. 
            upsample_factor_fft (int): The multiplication factor by which to
                upsample the fft'd data, this is used for subpixel image
                registration. 
            template_range (Tuple[int, int]): If no template is provided, this
                is the range over the frames to compute the template from. For
                example, `template_range = (10, 50)` would compute an initial 
                template from the frames in the range [10, 20).
            spatial_bandpass (bool): Whether to perform a spatial bandpass or
                not before finding the shifts. This does not effect the original
                data and can lead to better performance. 
            sigma (Tuple[float, float]): The sigma values for the spatial
                bandpass fiter.
            add_min (bool): Whether or not to add the minimum value when
                computing the rigid shifts. This can lead to better performance
                because the rigid shifts are found through the max of the
                absolute value of the phase correlation. So negative values can
                be troublesome in this sense.
            parallel_backend (str): The backend to use for motion correction. It
                was `loky` by default, but memory leaks became an issue and so
                now `threading` is recommended. Hopefuly the memory leak bug can
                be fixed in future updates. 
            n_jobs (int): The number of jobs for the parallel option. 
            verbose (bool): Whether or not to show the TQDM loading bars.

        """
        self.max_shifts = max_shifts
        self.upsample_factor_fft = upsample_factor_fft
        self.add_min = add_min
        self.template_range = template_range
        self.spatial_bandpass = spatial_bandpass
        self.sigma = sigma
        self.parallel_backend = parallel_backend
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.rigid_shifts = []
        self.template = None
        self._is_fit = False
        self._min = None

    def _init_min(self, frames):
        if self._min is None:
            if self.add_min:
                self._min = frames.min()
            else:
                self._min = 0.0

    def get_template(self, frames):
        l, r = self.template_range

        if (l < 0) or (r > len(frames)) or (l > r):
            raise ValueError("'template_range' was set incorrectly.")

        # Set the minimum value if it needs to be added.
        self._init_min(frames)

        # Get an initial template estimation that will be later updated with
        # rigid motion correction.
        self.template = np.mean(frames[l : r], axis = 0) + self._min
        if self.spatial_bandpass:
            self.template = gaussian_bandpass_filter(
                self.template, sigma = self.sigma
            )

        # Compute the first set of rigid shifts and temporarily store them.
        logger.info(f"Estimating template rigid shifts with {self.n_jobs} job(s).")
        with Parallel(n_jobs = self.n_jobs, backend = self.parallel_backend) as parallel:
            _rigid_shifts = parallel(
                delayed(self._get_shift)(frames[i]) for i in tqdm(range(l, r), disable = not self.verbose)
            )

        # Reset the template to only include a running mean of the motion
        # corrected frames. This gives a most robust estimate.
        self.template = np.zeros_like(self.template)
        logger.info(f"Computing template via a rolling mean.")
        for i in tqdm(range(r - l), disable = not self.verbose):
            frame = frames[l + i].copy().astype(np.float32) + self._min
            if self.spatial_bandpass:
                frame = gaussian_bandpass_filter(
                    frame, sigma = self.sigma
                )

            frame = apply_rigid_shift(frame, _rigid_shifts[i])

            # Update the template via rolling mean. This is significantly more
            # efficient than storing all of the frames in memory and using a
            # median, though likely gives a worse template. In my testing, this
            # method works quite well. The second, non-commented method, is
            # probably more numerically stable
            #self.template = ((self.template * i) + frame) / (i + 1)
            self.template = (self.template * (i / (i + 1))) + (frame / (i + 1))

        return self.template

    def fit(self, frames, template = None):
        if frames.ndim != 3:
            raise ValueError(
                "The `frames` array must have 3 dimensions and be of shape "
                "(n_frames, height, width)."
            )

        # It is probably bad to add min directly to `frames` because we would
        # be altering the original stack. Probably compute the min then add it
        # to each individual frame copy while computing the rigid shifts.
        self._init_min(frames)

        # If the template is None, compute it.
        self.template = template
        if self.template is None:
            logger.info(
                "No template was provided, computing the template with rigid "
                "motion correction."
            )
            self.template = self.get_template(frames)

        logger.info(f"Performing rigid motion correction with {self.n_jobs} job(s).")
        with Parallel(n_jobs = self.n_jobs, backend = self.parallel_backend) as parallel:
            self.rigid_shifts = parallel(
                delayed(self._get_shift)(frames[i]) for i in tqdm(range(len(frames)), disable = not self.verbose)
            )

        self._is_fit = True
        return self

    def _get_shift(self, image):
        """Modified, and private, version of 'get_rigid_shift' that encorperates
        optional spatial filtering. See `RigidMotionCorrection` for the argument 
        descriptions. 

        """
        image = image.astype(np.float32) + self._min

        if self.spatial_bandpass:
            image = gaussian_bandpass_filter(
                image, sigma = self.sigma
            )

        shift = get_rigid_shift(
            image = image, 
            template = self.template,
            upsample_factor = self.upsample_factor_fft,
            max_shifts = self.max_shifts,
        )

        return shift

    def transform(self, frames):
        if frames.ndim != 3:
            raise ValueError(
                "The `frames` array must have 3 dimensions of shape "
                "(n_frames, height, width)."
            )

        if len(frames) != len(self.rigid_shifts):
            raise ValueError(
                "`frames` and `rigid_shifts` must have the same length."
            )

        if not self._is_fit:
            raise NotFittedError(
                "`MotionCorrect` has not yet been fit. Run `fit` with the "
                "appropriate arguments before calling `transform`."
            )

        logger.info("Applying shifts.")
        for i in tqdm(range(len(frames)), disable = not self.verbose):
            frames[i] = apply_rigid_shift(frames[i], self.rigid_shifts[i])

        return frames
        
    def fit_transform(self, frames, template = None):
        self.fit(frames, template)
        return self.transform(frames)


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    """
    pass


