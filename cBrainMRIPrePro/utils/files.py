# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE 
| Created on: Nov 23, 2020
"""
import logging
import os
from typing import Tuple

import SimpleITK as sitk
import numpy as np

logger = logging.getLogger(__name__)


def check_isdir(input_dir: str) -> str:
    """
    Check if a directory exist.

    :param input_dir: string of the path of the input directory.
    :return: string if exist, else raise NotADirectoryError.
    """
    if os.path.isdir(input_dir):
        return input_dir
    else:
        raise NotADirectoryError(input_dir)


def check_is_nii_exist(input_file_path: str) -> str:
    """
    Check if a directory exist.

    :param input_file_path: string of the path of the nii or nii.gz.
    :return: string if exist, else raise Error.
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"{input_file_path} was not found, check if it's a valid file path")

    pth, fnm, ext = split_filename(input_file_path)
    if ext not in [".nii", ".nii.gz"]:
        raise FileExistsError(f"extension of {input_file_path} need to be '.nii' or '.nii.gz'")
    return input_file_path


def safe_file_name(file_name: str) -> str:
    """
    Remove any potentially dangerous or confusing characters from
    the file name by mapping them to reasonable substitutes.

    :param file_name: name of the file.
    :return: name of the file corrected.
    """
    underscores = r"""+`~!?@#$%^&*(){}[]/=\|<>,.":' """
    safe_name = ""
    for c in file_name:
        if c in underscores:
            c = "_"
        safe_name += c
    return safe_name


def split_filename(file_name: str) -> Tuple[str, str, str]:
    """
    Split file_name into folder path name, basename, and extension name.

    :param file_name: full path
    :return: path name, basename, extension name
    """
    pth = os.path.dirname(file_name)
    f_name = os.path.basename(file_name)

    ext = None
    for special_ext in ['.nii.gz']:
        ext_len = len(special_ext)
        if f_name[-ext_len:].lower() == special_ext:
            ext = f_name[-ext_len:]
            f_name = f_name[:-ext_len] if len(f_name) > ext_len else ''
            break
    if not ext:
        f_name, ext = os.path.splitext(f_name)
    return pth, f_name, ext


def load_nifty_volume_as_array(input_path_file: str) -> Tuple[np.ndarray, Tuple[Tuple, Tuple, Tuple]]:
    """
    Load nifty image into numpy array [z,y,x] axis order.
    The output array shape is like [Depth, Height, Width].

    :param input_path_file: input path file, should be '*.nii' or '*.nii.gz'
    :return: a numpy data array, (with header)
    """
    img = sitk.ReadImage(input_path_file)
    data = sitk.GetArrayFromImage(img)

    origin, spacing, direction = img.GetOrigin(), img.GetSpacing(), img.GetDirection()
    return data, (origin, spacing, direction)


def save_to_nii(im: np.ndarray, header: (tuple, tuple, tuple), output_dir: str, filename: str, mode: str = "image",
                gzip: bool = True) -> None:
    """
    Save numpy array to nii.gz format to submit.

    :param im: array numpy
    :param header: header metadata (origin, spacing, direction).
    :param output_dir: Output directory.
    :param filename: Filename of the output file.
    :param mode: save as 'image' or 'label'
    :param gzip: zip nii (ie, nii.gz)
    """
    origin, spacing, direction = header
    if mode == "label":
        img = sitk.GetImageFromArray(im.astype(np.uint8))
    else:
        img = sitk.GetImageFromArray(im.astype(np.float32))
    img.SetOrigin(origin), img.SetSpacing(spacing), img.SetDirection(direction)

    if gzip:
        ext = ".nii.gz"
    else:
        ext = ".nii"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    sitk.WriteImage(img, os.path.join(output_dir, filename) + ext)
