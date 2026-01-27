import warnings
from collections.abc import Callable, Sequence, Iterable
from itertools import chain
from math import ceil
from typing import Any

import numpy as np
import torch


def issequenceiterable(obj: Any) -> bool:
    try:
        if hasattr(obj, "ndim") and obj.ndim == 0:
            return False  # a 0-d tensor is not iterable
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

def ensure_tuple(vals: Any, wrap_array: bool = False) -> tuple:
    if wrap_array and isinstance(vals, (np.ndarray, torch.Tensor)):
        return (vals,)
    return tuple(vals) if issequenceiterable(vals) else (vals,)

def ensure_tuple_rep(tup: Any, dim: int) -> tuple[Any, ...]:

    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")

def fall_back_tuple(
    user_provided: Any, default: Any, func: Callable = lambda x: x and x > 0
) -> tuple[Any, ...]:
 
    ndim = len(default)
    user = ensure_tuple_rep(user_provided, ndim)
    return tuple(  # use the default values if user provided is not valid
        user_c if func(user_c) else default_c for default_c, user_c in zip(default, user)
    )

def correct_crop_centers(
    centers: list[int],
    spatial_size: Sequence[int] | int,
    label_spatial_shape: Sequence[int],
    allow_smaller: bool = False,
) -> tuple[Any]:
    """
    Utility to correct the crop center if the crop size and centers are not compatible with the image size.

    Args:
        centers: pre-computed crop centers of every dim, will correct based on the valid region.
        spatial_size: spatial size of the ROIs to be sampled.
        label_spatial_shape: spatial shape of the original label data to compare with ROI.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    """
    spatial_size = fall_back_tuple(spatial_size, default=label_spatial_shape)
    if any(np.subtract(label_spatial_shape, spatial_size) < 0):
        if not allow_smaller:
            raise ValueError(
                "The size of the proposed random crop ROI is larger than the image size, "
                f"got ROI size {spatial_size} and label image size {label_spatial_shape} respectively."
            )
        spatial_size = tuple(min(l, s) for l, s in zip(label_spatial_shape, spatial_size))

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size // np.array(2)).astype(int)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i, valid_s in enumerate(valid_start):
        # need this because np.random.randint does not work with same start and end
        if valid_s == valid_end[i]:
            valid_end[i] += 1
    valid_centers = []
    for c, v_s, v_e in zip(centers, valid_start, valid_end):
        center_i = min(max(c, v_s), v_e - 1)
        valid_centers.append(int(center_i))
    return ensure_tuple(valid_centers)

def generate_pos_neg_label_crop_centers(
    spatial_size: Sequence[int] | int,
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: Sequence[int],
    bg_indices: Sequence[int],
    rand_state: np.random.RandomState | None = None,
    allow_smaller: bool = False,
) -> tuple[tuple]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).

    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.
    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    centers = []
    labels = []
    fg_indices = np.asarray(fg_indices) if isinstance(fg_indices, Sequence) else fg_indices
    bg_indices = np.asarray(bg_indices) if isinstance(bg_indices, Sequence) else bg_indices
    if len(fg_indices) == 0 and len(bg_indices) == 0:
        raise ValueError("No sampling location available.")

    if len(fg_indices) == 0 or len(bg_indices) == 0:
        pos_ratio = 0 if len(fg_indices) == 0 else 1
        warnings.warn(
            f"Num foregrounds {len(fg_indices)}, Num backgrounds {len(bg_indices)}, "
            f"unable to generate class balanced samples, setting `pos_ratio` to {pos_ratio}."
        )
    
    if isinstance(spatial_size[0], int):
        print("Expand!!")
        spatial_size = [spatial_size] * num_samples

    for _, shape in enumerate(spatial_size):
        (indices_to_use, label) = (fg_indices, 1) if rand_state.rand() < pos_ratio else (bg_indices, 0)
        random_int = rand_state.randint(len(indices_to_use))
        idx = indices_to_use[random_int]
        center = np.array(np.unravel_index(idx, label_spatial_shape)).T.tolist()
        # shift center to range of valid centers
        labels.append(label)
        centers.append(correct_crop_centers(center, shape, label_spatial_shape, allow_smaller))

    return ensure_tuple(centers), ensure_tuple(labels)

def generate_label_classes_crop_centers(
    spatial_size: Sequence[int] | int,
    num_samples: int,
    label_spatial_shape: Sequence[int],
    indices: Sequence[Any],
    ratios: list[float | int] | None = None,
    rand_state: np.random.RandomState | None = None,
    allow_smaller: bool = False,
    warn: bool = True,
) -> tuple[tuple]:
    """
    Generate valid sample locations based on the specified ratios of label classes.
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        indices: sequence of pre-computed foreground indices of every class in 1 dimension.
        ratios: ratios of every class in the label to generate crop centers, including background class.
            if None, every class will have the same ratio to generate crop centers.
        rand_state: numpy randomState object to align with other modules.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
        warn: if `True` prints a warning if a class is not present in the label.

    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    if num_samples < 1:
        raise ValueError(f"num_samples must be an int number and greater than 0, got {num_samples}.")
    ratios_: list[float | int] = list(ensure_tuple([1] * len(indices) if ratios is None else ratios))
    if len(ratios_) != len(indices):
        raise ValueError(
            f"random crop ratios must match the number of indices of classes, got {len(ratios_)} and {len(indices)}."
        )
    if any(i < 0 for i in ratios_):
        raise ValueError(f"ratios should not contain negative number, got {ratios_}.")

    for i, array in enumerate(indices):
        if len(array) == 0:
            if ratios_[i] != 0:
                ratios_[i] = 0
                if warn:
                    warnings.warn(
                        f"no available indices of class {i} to crop, setting the crop ratio of this class to zero."
                    )

    centers = []
    classes = rand_state.choice(len(ratios_), size=num_samples, p=np.asarray(ratios_) / np.sum(ratios_))
    for i in classes:
        # randomly select the indices of a class based on the ratios
        indices_to_use = indices[i]
        random_int = rand_state.randint(len(indices_to_use))
        center = unravel_index(indices_to_use[random_int], label_spatial_shape, order='C').tolist()
        # shift center to range of valid centers
        centers.append(correct_crop_centers(center, spatial_size, label_spatial_shape, allow_smaller))

    return ensure_tuple(centers)


def map_classes_to_indices(label_img, 
                            labels: Sequence[int] | int,
                            one_hot: bool = False, 
                            max_samples_per_class: int | None = None):
    indicies: list[np.ndarray] = []
    # IF label is one-hot encoded
    if isinstance(labels, int):
        labels = list(range(labels))
    
    for label in labels:
        if one_hot is True:
            label_flat = np.ravel(label_img[label]).astype(bool)
        else:
            label_flat = np.ravel(label_img == label, order='C')
        print("Label_flat shape", label_flat.shape)
        cls_indices = np.nonzero(label_flat)[0]
        if max_samples_per_class and len(cls_indices) > max_samples_per_class and len(cls_indices) > 1:
            sample_id = np.round(np.linspace(0, len(cls_indices) - 1, max_samples_per_class)).astype(int)
            indicies.append(cls_indices[sample_id])
        else:
            indicies.append(cls_indices)
    
    return indicies

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    image = np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1], size=[256, 512])
    # image = cv2.resize(image, [512, 512], cv2.INTER_NEAREST)
    print(np.unique(image, return_counts=True))
    indicies = map_classes_to_indices(image, [0, 1], one_hot=False)
    centers = generate_pos_neg_label_crop_centers(  [32, 32], 
                                                    num_samples=32, 
                                                    pos_ratio=0.1, 
                                                    label_spatial_shape=image.shape,
                                                    fg_indices=indicies[1], 
                                                    bg_indices=indicies[0], 
                                                    rand_state=np.random.RandomState(seed=12), 
                                                    allow_smaller=False)
    print(len(centers), centers)
    centers = np.array(centers)
    
