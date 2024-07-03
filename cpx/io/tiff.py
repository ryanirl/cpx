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
import tifffile as tiff
import numpy as np


def load_tiff(input_file, dtype = None):
    """ """
    # Previously, 'tiff.imread' was used. It turns out this is extremely
    # inefficient and that writing and reading using TiffFile and TiffWriter
    # is magnitudes quicker. 
    with tiff.TiffFile(input_file) as tiff_file:
        num_frames = len(tiff_file.pages)

        if len(tiff_file.pages) < 1:
            raise ValueError(
                f"The tiff file '{input_file}' has less than one page. Check "
                f"to make sure it's saved properly."
            )

        if dtype is None:
            dtype = tiff_file.pages[0].dtype

        shape = tiff_file.pages[0].shape
        output_shape = (num_frames, *shape)

        # Pre-allocate the array to ensure memory efficiency and safety. 
        output_array = np.zeros(output_shape, dtype = dtype)
        for i in tqdm(range(len(tiff_file.pages))):
            output_array[i] = tiff_file.pages[i].asarray().astype(dtype)

    return output_array


def write_tiff(output_file, array, bigtiff = True, **kwargs):
    """ """
    with tiff.TiffWriter(output_file, bigtiff = bigtiff, **kwargs) as tiff_file:
        for i in tqdm(range(len(array))):
            tiff_file.write(array[i])


def load_tiff_page(input_file, page = 0):
    """Loads and returns the first tiff image in a stack."""
    with tiff.TiffFile(input_file) as tiff_file:
        if len(tiff_file.pages) <= page:
            raise ValueError(
                f"The tiff file '{input_file}' has less pages than '{page}'."
            )

        image = tiff_file.pages[page].asarray()
    return image


