# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Nov 23, 2020
"""
import logging
from typing import Tuple

import numpy as np
from numba import njit, prange
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.WARNING)


def _handle_zeros_in_scale(scale, copy=True):
    """
    Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    """

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


@njit(parallel=True)
def fill_mask(mask_arr: np.ndarray) -> np.ndarray:
    """
    Fill a 3D mask array. Useful when use a threshold function and need to fill hole

    :param mask_arr: mask to fill
    :return: mask filled
    """
    assert mask_arr.ndim == 3, "Mask to fill need to be a 3d array"
    for z in prange(0, mask_arr.shape[0]):  # we use np convention -> z,y,x
        for x in prange(0, mask_arr.shape[2]):
            if np.max(mask_arr[z, :, x]) == 1:
                a0 = mask_arr.shape[1] - 1
                b0 = 0
                while mask_arr[z, a0, x] == 0:
                    if a0 != 0:
                        a0 = a0 - 1  # Top of the data. Above it is zero.

                while mask_arr[z, b0, x] == 0:
                    if b0 != mask_arr.shape[1] - 1:
                        b0 = b0 + 1  # Bottom of the data. Below it is zero.
                for k in prange(b0, a0 + 1):
                    mask_arr[z, k, x] = 1
        for y in prange(0, mask_arr.shape[1]):
            if np.max(mask_arr[z, y, :]) == 1:
                c0 = mask_arr.shape[2] - 1
                d0 = 0
                while mask_arr[z, y, c0] == 0:
                    if c0 != 0:
                        c0 = c0 - 1  # Top of the data. Above it is zero.

                while mask_arr[z, y, d0] == 0:
                    if d0 != mask_arr.shape[1] - 1:
                        d0 = d0 + 1  # Bottom of the data. Below it is zero.
                for j in prange(d0, c0 + 1):
                    mask_arr[z, y, j] = 1
    return mask_arr


def get_mask(input_array: np.ndarray) -> np.ndarray:
    """
    Get a (head) mask. Based on Otsu threshold and noise reduced. Then result mask is holes filled.

    :param input_array: input image array
    :return: binary head mask
    """
    thresh = threshold_otsu(input_array)
    otsu_mask = input_array > thresh
    noise_reduced = remove_small_objects(otsu_mask, 10, )
    head_mask = fill_mask(noise_reduced.astype(np.uint8))
    return head_mask


def min_max_scaling(input_array: np.ndarray, scaling_range: Tuple[int, int] = (10, 100)) \
        -> Tuple[np.ndarray, float, float]:
    """
    Transform image input pixels/voxels to a given range. (type min-max scaler)

    :param input_array: input image array
    :param scaling_range: min, max = feature_range
    :return: image_scaled, min, scale
    """
    image_scaled = np.copy(input_array).astype(np.float32)
    scale_ = ((scaling_range[1] - scaling_range[0]) / _handle_zeros_in_scale(
        np.max(input_array) - np.min(input_array))).astype(np.float32)
    min_ = scaling_range[0] - np.min(input_array) * scale_
    image_scaled *= scale_
    image_scaled += min_
    return image_scaled, min_, scale_


def invert_min_max_scaling(input_array_scaled: np.ndarray, scale_: float, min_: float) -> np.ndarray:
    """
    Invert min max scaling

    :param input_array_scaled: input image array scaled
    :param scale_: Per pixels/voxels relative scaling of the data.
    :param min_: Per pixels/voxels minimum seen in the data
    :return: input image array unscaled
    """
    input_array_scaled = np.copy(input_array_scaled).astype(np.float32)
    input_array_scaled -= min_
    input_array_scaled /= scale_
    return input_array_scaled


def zscore_normalize(input_array: np.ndarray, scaling_factor: int = 1, mask: np.ndarray = None):
    """
    Function to normalize array with Z-Score. Normalize a target image by subtracting the mean of the whole brain
    and dividing by the standard deviation.

    :param input_array: input array image to normalize
    :param scaling_factor: scaling factor to apply to normalization
    :param mask: input mask where to apply the normalization. If not provided default will be in non zero value
    :return: input array normalize
    """
    if mask is None:
        mask = input_array > 0
    logical_mask = mask == 1  # force the mask to be logical type
    mean = input_array[logical_mask].mean()
    std = input_array[logical_mask].std()
    normalized_input_array = (((input_array - mean) / std) * scaling_factor) * logical_mask
    return normalized_input_array
