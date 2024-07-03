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

import pathlib
import os

from ._exceptions import *


def validate_file_exists(file_path):
    """Validate whether the file exists at the given path.
    
    Args:
        file_path (str): The path to the file.
        
    Returns:
        None

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")


def validate_file_not_exists(file_path):
    """Validate that the file does not exists at the given path. Useful for
    validating output files.
    
    Args:
        file_path (str): The path to the file.
        
    Returns:
        None

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    if os.path.exists(file_path):
        raise FileExistsError(f"File already exists: '{file_path}'")


def validate_parent_directory(path):
    """Validate whether the parent directory of the given file path exists.
    
    Args:
        path (str): The path to the file.

    Returns:
        None
        
    Raises:
        ParentDirectoryNotFoundError: If the parent directory does not exist.

    """
    # Using pathlib here fixes some edge cases with 'os.dirname()'.
    parent_directory = pathlib.Path(path).parent.absolute()
    
    # Handle the case when the parent directory is the current directory and
    # thus an empty string. Note: Since switching to pathlib, this has become a
    # pointless check.
    if not parent_directory:
        return
    
    if not os.path.exists(parent_directory):
        raise ParentDirectoryNotFoundError(
            f"Parent directory '{parent_directory}' does not exist."
        )


def validate_file_extension(file_path, allowed_extensions = None):
    """Validate whether the file path has one of the allowed extensions.
    
    Args:
        file_path (str): The path to the file.
        allowed_extensions (list): List of allowed extensions (e.g., 
            ['.tiff', '.tif'] for TIFF files).

    Returns:
        None
        
    Raises:
        InvalidFileExtensionError: If the file does not have an allowed
            extension.

    """
    if allowed_extensions is None:
        return 

    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() not in allowed_extensions:
        ext_strs = [f"'{ext}'" for ext in allowed_extensions]
        allowed_extensions_str = ", ".join(ext_strs)

        raise InvalidFileExtensionError(
            f"Invalid file extension: '{file_extension}'. Only "
            f"{allowed_extensions_str} extensions are allowed."

        )



