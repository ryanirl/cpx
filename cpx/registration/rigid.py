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
import cv2

from ._numerics import phase_cross_correlation 


def get_rigid_shift(
    image, 
    template, 
    upsample_factor = 1, 
    max_shifts = (10, 10), 
    space = "real"
):
    """An alias for 'phase_cross_correlation', see 'cpx/registration/numerics.py' 
    for more information. The recommended usage is as follows:

        >>> rigid_shift = get_rigid_shift(image, template)
        >>> registered_image = apply_rigid_shift(image, rigid_shift) 

    """
    return phase_cross_correlation(
        image = image, 
        template = template, 
        upsample_factor = upsample_factor, 
        max_shift = max_shifts,
        space = space,
    )


def apply_rigid_shift(
    image,
    rigid_shift,
    interp_mode = cv2.INTER_LINEAR,
    border_mode = cv2.BORDER_REFLECT, 
    border_value = 0
):
    """ """
    h, w = image.shape
    m = np.float32([
        [1, 0, -rigid_shift[1]], 
        [0, 1, -rigid_shift[0]]
    ])
    
    # I was using 'scipy.ndimage.affine_transform()', but this turns out to be
    # significantly slower than 'cv2.warpAffine()'. I need to look into the
    # differences because I trust scipy's version more. I don't know why cv2
    # would be so significantly faster.
    image = cv2.warpAffine(
        image, m, (w, h),
        flags = interp_mode,
        borderMode = border_mode,
        borderValue = border_value
    )
    return image


