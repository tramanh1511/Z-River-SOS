import numpy as np
import cv2
import scipy

def negative_field(logit_map, distance=15, beta=0.5, alpha=0.05):
    if isinstance(distance, int):
        distance = (distance, distance)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, distance)
    dilated_mask = cv2.morphologyEx(cv2.dilate((logit_map > beta).astype(int).astype(np.uint8), kernel, iterations=5), cv2.MORPH_GRADIENT, kernel)
    possible = scipy.ndimage.binary_fill_holes(np.where(logit_map > alpha, 1, 0)).astype(int).astype(np.uint8)
    dilated_possible = cv2.dilate(possible, kernel, iterations=3)
    negative_field = dilated_mask - dilated_mask * dilated_possible
    return negative_field


def grid_sampling(mask, grid=8, alpha=0.1):
    if isinstance(grid, int):
        grid = (grid, grid)
    # We cannot sampling on grid
    x, y = np.linspace(0, 1, grid[0])[:-1], np.linspace(0, 1, grid[1])[:-1]
    patch_size = np.array(mask.shape[:2]) // grid
    mesh = np.floor(np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2) * mask.shape[:2]).astype(int)
    def sample(src):
        dst = src + patch_size
        if np.mean(mask[src[0]:dst[0],src[1]:dst[1]]) < alpha:
            return [-1, -1]
        possible = np.array(np.where(mask[src[0]:dst[0],src[1]:dst[1]] > 0)).T
        return possible[np.random.randint(0, high=possible.shape[0])] + src
    output = np.apply_along_axis(sample, 1, mesh)
    return output[output[:, 0] > 0]