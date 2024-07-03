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

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np


def plot(signal, *, macro = "", figsize = (12, 2), **kwargs):
    plt.figure(figsize = figsize)
    plt.plot(signal, **kwargs)
    exec(macro)
    plt.show()


def imshow(image, *, macro = "", cmap = "magma", figsize = (18, 12), **kwargs):
    plt.figure(figsize = figsize)
    plt.imshow(image, cmap = cmap, **kwargs)
    exec(macro)
    plt.show()
    

def save_image(filename, array, show = True, cmap = "magma"):
    fig, ax = plt.subplots(
        figsize = (
            array.shape[1] / 100, 
            array.shape[0] / 100
        ), 
        dpi = 100
    )

    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0)

    ax.imshow(array, cmap = cmap)

    plt.savefig(
        filename, 
        transparent = False, 
        bbox_inches = "tight", 
        pad_inches = 0
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


def grayscale_to_rgb(image: np.ndarray, cmap: str = "magma") -> np.ndarray:
    """Converts a grayscale 2D image to a 3D colormaped image. 
    
    Args:
        image (np.ndarray): The input image of ndim == 2.
        cmap (str): A string representing the matplotlib colormap.
        
    Returns:
        np.ndarray: The cmap colored image.
        
    """
    import matplotlib as mpl

    if image.ndim != 2:
        raise ValueError("'image' must have ndim == 2.")
        
    # Return the colored array in uint8 format.
    return (mpl.colormaps[cmap](image) * 255).astype(np.uint8)


def write_video(video_dir: str, video: np.ndarray, fps: int = 15) -> None:
    # Defining this here could lead to post-validation crashing if I run
    # this function in CLI-based processes. 
    import imageio

    if video.dtype != np.uint8:
        minn = video.min()
        maxx = video.max()

    writer = imageio.get_writer(video_dir, fps = fps)
    for image in tqdm(video):
        if image.dtype != np.uint8:
            image = (image - minn) / (maxx - minn)
            image = (image * 255).astype(np.uint8)

        writer.append_data(image)
    writer.close()


def write_video_rgb(
    video_dir: str, 
    video: np.ndarray, 
    *,
    cmap: str = "magma", 
    fps: int = 15
) -> None:
    # Defining this here could lead to post-validation crashing if I run
    # this function in CLI-based processes. 
    import imageio
    
    maxx = video.max()
    minn = video.min()

    writer = imageio.get_writer(video_dir, fps = fps)
    for image in tqdm(video):
        image = image - minn / (maxx - minn)
        image = grayscale_to_rgb(image, cmap)

        writer.append_data(image)
    writer.close()


