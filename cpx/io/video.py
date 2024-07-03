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
import warnings
import cv2

from typing import Optional
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Any

from .utils import to_grayscale
from .utils import get_video_metadata


def video_writer_fourcc(codec: str) -> int:
    """Custom implementation of `cv2.VideoWriter_fourcc()` to ensure version
    compatibility.
    
    Args:
        codec (str): FourCC codec string (e.g., 'XVID', 'MJPG', 'MP4V', etc.).

    Returns:
        int: FourCC codec value.

    """
    if len(codec) != 4:
        raise ValueError("Codec must be a four-character string.")
    
    # Convert characters to ASCII codes and concatenate
    fourcc = (
        (ord(codec[0]) <<  0) | 
        (ord(codec[1]) <<  8) | 
        (ord(codec[2]) << 16) | 
        (ord(codec[3]) << 24)
    )
    return fourcc


def load_video(
    input_file: str, 
    dtype: Optional[str] = None, 
    return_metadata: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Loads a video into a numpy array. Notice that it is automatically
    converted to grayscale. 

    """
    reader = cv2.VideoCapture(input_file)
    if not reader.isOpened():
        raise ValueError(
            f"The cv2 VideoCapture cannot read the input file '{input_file}'."
        )

    metadata = get_video_metadata(reader)

    width = metadata["width"]
    height = metadata["height"]
    n_frames = metadata["frame_count"]
    output_shape = (n_frames, height, width)

    if n_frames < 1:
        raise ValueError(
            f"The video file '{input_file}' has less than one page."
        )

    if dtype is None:
        dtype = "uint8"

    # Pre-allocate the array to ensure memory efficiency and safety. 
    output_array = np.zeros(output_shape, dtype = dtype)
    for i in tqdm(range(n_frames)):
        ret, image = reader.read()

        if not ret:
            print("Exiting early.")
            break

        image = to_grayscale(image)
        image = image.astype(dtype)
        output_array[i] = image

    reader.release()

    if return_metadata:
        return output_array, metadata

    return output_array


def write_video(
    output_file: str, 
    frames: np.ndarray, 
    *,
    fps: int = 30, 
    codec: str = "FFV1", 
    is_color: bool = False
) -> None:
    """
    """
    n_frames, height, width = frames.shape

    # Initialize video writer.
    fourcc = video_writer_fourcc(codec)
    writer = cv2.VideoWriter(
        output_file, fourcc, fps, (width, height), isColor = is_color
    )

    scale = False
    if frames.dtype != "uint8":
        warnings.warn(
            "The input video stack is not of type 'uint8', scaling data "
            "between 0 and 255 then converting. This may result in unexpected "
            "behavior."
        )
        minn = frames.min()
        maxx = frames.max()
        scale = True

    # Write frames to video.
    for i in tqdm(range(n_frames)):
        image = frames[i]
        if scale:
            image = (image - minn) / (maxx - minn)
            image = (image * 255).astype("uint8")

        writer.write(image)
    writer.release()


