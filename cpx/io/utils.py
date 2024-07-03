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
import os

from typing import Dict


def attach_prefix_to_filename(filename: str, prefix: str) -> str:
    """Attaches a prefix to the given filename.

    Args:
        filename (str): The original filename.
        prefix (str): The prefix to attach.

    Returns:
        str: The new filename with the attached prefix.

    """
    directory, filename = os.path.split(filename)
    return os.path.join(directory, prefix + filename)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale. 

    Args:
        image (np.ndarray): The RGB image to be converted to grayscale.
        
    Returns:
        np.ndarray: The grayscale 2D image.
    
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_video_metadata(cap: cv2.VideoCapture) -> Dict:
    """Return a dictionary of metadata for the cv2 VideoCapture object.

    Args:
        cap (cv2.VideoCapture): An instantiated VideoCapture object.

    Returns:
        Dict: A dictionary of metadata.

    """
    metadata = {
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    }

    return metadata


