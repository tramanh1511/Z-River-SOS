import numpy as np
import cv2
import scipy
import skimage

def smooth_mask(mask, kernel_size, sigma):
    output = np.zeros_like(mask)
    kernel = cv2.getGaussianKernel(kernel_size, sigma).squeeze()
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    refined = [np.stack([np.convolve(kernel, cnt[:, 0, i], mode='valid') for i in range(cnt.shape[-1])], axis=-1).round().astype(np.int32)[:, None, :] for cnt in cnts]
    cv2.drawContours(output, refined, -1, 1, -1)
    return output

def smooth_graph(cnt, kernel_size, sigma):
    kernel = cv2.getGaussianKernel(kernel_size, sigma).squeeze()
    return np.stack([np.convolve(kernel, cnt[:, i], mode='valid') for i in range(cnt.shape[-1])], axis=-1).round().astype(int)

def morphology_closing(mask, kernel, iterations=1):
    temp = cv2.dilate(mask, kernel, iterations)
    return cv2.erode(temp, kernel, iterations)

def getLargestCC(segmentation):
    labels = skimage.measure.label(segmentation, connectivity=2)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def prune(mask, min_size=5):
    output = np.zeros(mask.shape, np.uint8)
    _, label_im = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
    labels, counts = np.unique(label_im, return_counts=True)
    for label, count in zip(labels[1:], counts[1:]):
        im = label_im == label
        if count >= min_size:
            output = cv2.bitwise_or(output, im.astype(np.uint8))
    return output

def morphology_thinning(mask, return_weight=False):
  #thinning word into a line
  # Structuring Element
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
  close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
  weight = np.zeros(mask.shape, dtype=np.uint8)
  # early stopping
  if cv2.countNonZero(cv2.erode(mask,kernel)) == 0:
    if return_weight: 
      return mask, weight
    return mask

  # Create an empty output image to hold values
  thin = np.zeros(mask.shape,dtype='uint8')
  # Loop until erosion leads to an empty set
  while cv2.countNonZero(mask)!= 0:
    # Erosion
    erode = cv2.erode(mask,kernel)
    # Opening on eroded image
    opened = cv2.morphologyEx(erode,cv2.MORPH_OPEN, close_kernel)
    # Subtract these two
    subset = erode - opened
    # Union of all previous sets
    thin = cv2.bitwise_or(subset,thin)
    # Keep the cummulative for weighting
    weight += thin
    # Set the eroded image for next iteration
    mask = erode.copy()
  
  if not return_weight:
    return thin
  else:
    return thin, weight


