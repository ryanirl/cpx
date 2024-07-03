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
import psutil

from typing import Tuple
from typing import Any


def flush_if_memmap(array: Any) -> None:
    """Checks if an array is a memory mapped array, then flushes the contents if
    so.

    Args:
        array (np.ndarray): The memory mapped array to flush the values to
            memory of.

    Return:
        None

    """
    if is_memmap(array):
        array.flush()


def is_memmap(array: Any) -> bool:
    """Checks if an object (particularly an array) is a memory mapped array.

    Args:
        array (np.ndarray): The array to check if it's memory mapped or not. 

    Returns:
        bool: 'True' if array is memory mapped, 'False' otherwise. 

    """
    try:
        return isinstance(array, np.memmap)
    except:
        return False


def get_memory_usage() -> Tuple[float, float]:
    """Returns values representing the total memory usage (RAM) in GB. 

    Args:
        None
    
    Returns:
        Tuple[float, float]: Two values representing the memory usage.

    """
    ram = psutil.virtual_memory()

    used  = ram.used / (1024 ** 3) 
    total = ram.total / (1024 ** 3)

    return used, total


