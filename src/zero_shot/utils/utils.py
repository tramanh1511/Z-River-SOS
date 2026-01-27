import numpy as np
import os
import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt

def plot_metric_distributions(per_sample_metrics, bins=20, out_dir=None, prefix="", show=False):
    """
    per_sample_metrics: dict from compute_per_sample_multilabel_metrics
    bins: number of histogram bins
    """
    for name, values in per_sample_metrics.items():
        values = np.asarray(values, dtype=float)
        values[np.isinf(values)] = 0
        plt.figure()
        plt.hist(values, bins=bins, density=True, align='right')
        plt.title(f"Distribution of {name}")
        plt.xlabel(name)
        plt.ylabel("Percentage")
        plt.grid(True)
        if out_dir is not None:
            plt.savefig(os.path.join(out_dir, f"{prefix}_Histogram_of_{name}"))
        if show:
            plt.show()
        else: 
            plt.close()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_bbox(bound, pt, patch_size):
    x = min(bound[1] - patch_size[1], max(0, pt[1] - patch_size[1] // 2))
    y = min(bound[0] - patch_size[0], max(0, pt[0] - patch_size[0] // 2))
    return x, y, patch_size[0], patch_size[1]

def save_gray(filename, image, cmap, invert=False, nonzero=True, output_size=None, verbose = False):
    if -1 in output_size:
        return False
    order = 1 if not invert else -1
    if nonzero:
        normed = image.copy()
        masked = image[image != 0]
        if len(masked) == 0: 
            print("Skip non-zero zero image")
            return False
        normed[normed != 0] = np.interp(masked, (masked.min(), masked.max()), (0, 1)[::order])
    else:
        normed = np.interp(image, (image.min(), image.max()), (0, 1)[::order])
    rgb_image = mpl.colormaps[cmap](normed)
    bgr_image = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) 
    if output_size is not None:
        bgr_image = cv2.resize(bgr_image, output_size)
    ret = cv2.imwrite(filename, bgr_image)
    if verbose:
        print(ret)

def save_grey(filename, image, verbose=False):
    normed = np.interp(image, (image.min(), image.max()), (0, 255)).astype(int).astype(np.uint8)
    ret = cv2.imwrite(filename, normed[..., None])
    if verbose:
        print(ret)