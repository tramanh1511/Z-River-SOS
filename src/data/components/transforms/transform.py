import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import multiprocessing
import tqdm

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.components.transforms.array import * 

def point_transform(points, M, homogeneous=False, transpose=False):  
  """Extension for point-wise transformation for corners"""

  if homogeneous is False: 
    padded = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=(1,))
  else:
    padded = points
  
  if transpose is True:
    padded[:, :2] = np.roll(padded[:, :2], 1, axis=1)
  transformed = np.vstack([M @ p for p in padded])
  
  if transpose is True:
    transformed[:, :2] = np.roll(transformed[:, :2], -1, axis=1)
  
  if homogeneous is True:
    return transformed
  else:
    return transformed[:, :2]

def norm_translate(h, w): 
  return np.array([[1, 0, - w / 2], [0, 1, - h / 2], [0, 0, 1]]), np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]])

def centric_kernel(h, w, mat):
  translate, inv_translate = norm_translate(h, w)
  return inv_translate @ mat @ translate

def decentric_kernel(in_h, in_w, out_h, out_w, mat):
  in_trans, _ = norm_translate(in_h, in_w)
  _, inv_out_trans = norm_translate(out_h, out_w)
  return inv_out_trans @ mat @ in_trans

def transform_compose(header, 
                      shear_x, 
                      shear_y,  
                      scale_x, 
                      scale_y, 
                      flip_x, 
                      flip_y, 
                      angle, 
                      random_order):
  """ Shape transform wrapper by fore-matmul all transformation 
  """
  # shear transformation
  # Translate for center preservationc
  scale = np.array([  [1 + scale_x, 0, 0],
                      [0, 1 + scale_y, 0],
                      [0, 0, 1]])
  shear = np.array([ [1, shear_x, 0],
                     [shear_y, 1, 0],
                     [0, 0, 1]])
  # skew transformation
  flip = np.array([ [flip_x, 0, 0],
                     [0, flip_y, 0],
                     [0, 0, 1]])
  rotate = cv2.getRotationMatrix2D((0, 0), angle=-angle, scale=1.)
  rotate = np.pad(rotate, ((0, 1), (0, 0)), mode='constant', constant_values=(0,))
  rotate[2, 2] = 1
  adhoc = [rotate, shear, scale, flip]
  if random_order:
    random.shuffle(adhoc)
  transform =  adhoc[0] @ adhoc[1] @ adhoc[2]

  inv_transform = np.linalg.inv(transform)
  inv_transform = centric_kernel(header['h'], header['w'], inv_transform)
  # Determine croppad
  transformed_size = transformed_patch_size(header['h'], header['w'], inv_transform)
  output_size= [max(header['h'], transformed_size[0]), max(header['w'], transformed_size[1])]
  transform = decentric_kernel(transformed_size[0], transformed_size[1], output_size[0], output_size[1], transform)

  return (transform, transformed_size, output_size)

def centric_transform(img, transform, fit=True, min_h=0, min_w=0):
  translate, inv_translate = norm_translate(img.shape[0], img.shape[1])
  kernel = inv_translate @ transform @ translate
  h, w = img.shape[:2]
  if fit is True: 
    corner = np.zeros((4, 2))
    corner[1:3, 1] += h
    corner[2:4, 0] += w
    transformed_corner = point_transform(corner, kernel)
    size = (transformed_corner.max(axis=0) - transformed_corner.min(axis=0)).astype(int)
    h, w = max(min_h, size[0]), max(min_w, size[1])
    addon = np.where(transformed_corner.min(axis=0) < 0, -transformed_corner.min(axis=0), 0)
    kernel[:2, 2] += addon
  return cv2.warpPerspective(img, kernel, (h, w), borderMode=cv2.BORDER_REFLECT)

def transformed_patch_size(h, w, mat):
    corner = np.zeros((4, 2))
    corner[1:3, 1] += h
    corner[2:4, 0] += w
    transformed_corner = point_transform(corner, mat)
    # print(transformed_corner)
    size = transformed_corner.max(axis=0) - transformed_corner.min(axis=0) + 1
    return np.ceil(size).astype(int)

def random_cut(spatial_size, 
                spatial_label_size, 
                profile, 
                sample_size, 
                randomState: int | None = None, 
                num_workers=4):
    header = {'h':spatial_size[0], 'w':spatial_size[1]}
    generator = np.random.RandomState(randomState) if randomState is not None else np.random.RandomState.__self__
    augments = random_augment(header, compile=True, generator=generator, **profile)
    

def affine_cut(image, pt, augment):
    forward, backward = transform_compose(**augment)
    h, w = augment['h'], augment['w']
    # print(h, w)
    back_translate, back_inv_translate = norm_translate(h, w)
    back = back_inv_translate @ backward @ back_translate    
    # print(transformed_corner)
    size = transformed_patch_size(h, w, back)
    cut = (size.astype(int) // 2)[::-1]
    ref_image = image[pt[0] - cut[0]:pt[0] + cut[0],
                        pt[1] - cut[1]:pt[1] + cut[1]]
    # print("Cut image shape:", ref_image.shape)
    transformed = centric_transform(ref_image, forward, min_h=h,min_w=w)
    size = transformed.shape
    # print("Transformed shape & bbox:", size)
    output =  transformed[ (size[0] - h)//2:(size[0] + h)//2,
                            (size[1] - w)//2:(size[1] + w)//2].copy()
    # print("Output shape", output.shape) 
    return output

# Compose augmentation
to_range = lambda x: [-x, x] if isinstance(x, float) else x
def random_augment(generator, header, 
                    shear_range = 0, 
                    rotate_range = 0, 
                    scale_range = 0, 
                    sample_size = 0, 
                    flip_ratio = 0, 
                    deform_grid = 0, 
                    deform_range=0.2,
                    shuffle_ratio=0.1,
                    compile=False):
    augments = []
    deform_grid = [deform_grid]
    shear_range = to_range(shear_range)
    rotate_range = to_range(rotate_range)
    scale_range = to_range(scale_range)
    # xx = np.linspace(0, 1, deform_grid)
    # xx, xx = np.linspace()

    shear_x = generator.uniform(shear_range[0], shear_range[1], size=sample_size)
    shear_y = generator.uniform(shear_range[0], shear_range[1], size=sample_size)
    flip_x = generator.choice([-1, 1], p=[flip_ratio, 1 - flip_ratio], size=sample_size)
    flip_y = generator.choice([-1, 1], p=[flip_ratio, 1 - flip_ratio], size=sample_size)
    rotate = generator.uniform(rotate_range[0], rotate_range[1], size=sample_size)
    scale_x = generator.uniform(scale_range[0], scale_range[1], size=sample_size)
    scale_y = generator.uniform(scale_range[0], scale_range[1], size=sample_size)
    order = generator.choice([True, False], p=[shuffle_ratio, 1 - shuffle_ratio], size=sample_size)
    for i in tqdm.tqdm(range(sample_size), desc="Composing deformation", leave=False):
        if compile is False:
            augments.append((header, shear_x[i], shear_y[i], scale_x[i], scale_y[i], flip_x[i], flip_y[i], rotate[i], order[i]))
        else: 
            augments[i] = dict(zip(['forward', 'size'], transform_compose(header=header,
                                                                          shear_x=shear_x[i], 
                                                                          shear_y=shear_y[i], 
                                                                          scale_x=scale_x[i], 
                                                                          scale_y=scale_y[i],
                                                                          flip_x=flip_x[i],
                                                                          flip_y=flip_y[i],
                                                                          angle=rotate[i],
                                                                          random_order=order[i])))
    return augments

def transform(image, gnt, pts, header, aug, sample_size, num_workers=4):
    augments = random_augment(gnt, header, sample_size=sample_size, **aug)
    inputs = iter([(image, pt, aug) for pt, aug in zip(pts, augments)])
    with multiprocessing.Pool(processes=num_workers) as p:
        outputs = p.starmap(affine_cut, tqdm.tqdm(inputs, total=sample_size, desc="Augmentation process"))
    return outputs

def warp_transform(img, kernel, output_size, size):
  # print(output_size, size)
  # output =  cv2.warpPerspective(img, kernel, output_size[::-1], borderMode=cv2.BORDER_REFLECT)
  # print(output.shape, output_size)
  # print((output_size[0] - size[0]) // 2,  (output_size[0] + size[0]) // 2, (output_size[1] - size[1]) // 2 , (output_size[1] + size[1]) // 2)
  return cv2.warpPerspective(img, kernel, output_size[::-1], borderMode=cv2.BORDER_REFLECT)[(output_size[0] - size[0]) // 2 : (output_size[0] + size[0]) // 2,
                                                                                            (output_size[1] - size[1]) // 2 : (output_size[1] + size[1]) // 2, :]

def cut_centers_transform(image, 
                          gnt, 
                          fg_indices, 
                          bg_indices, 
                          header, 
                          aug, 
                          profiles=None, 
                          centers=None, 
                          sample_size=512, 
                          pos_ratio=0.1, 
                          num_workers=4):
    labels = None
    if profiles is None:
      augments = random_augment(gnt, header, sample_size=sample_size, **aug)
      augments = iter(augments)
      with multiprocessing.Pool(processes=num_workers) as p:
          profiles = p.starmap(transform_compose, tqdm.tqdm(augments, total=sample_size, desc="Augmentation generations", leave=False))
      patch_sizes = [profile[1] for profile in profiles]
      # print(len(profiles), profiles[0][1], profiles[0][0].dtype, profiles[0][0].shape)
      # print(len(patch_sizes))
      centers, labels = generate_pos_neg_label_crop_centers(  patch_sizes, 
                                                      num_samples=sample_size, 
                                                      pos_ratio=pos_ratio, 
                                                      label_spatial_shape=image.shape[:2],
                                                      fg_indices=fg_indices, 
                                                      bg_indices=bg_indices, 
                                                      rand_state=gnt, 
                                                      allow_smaller=False)
    else:   
      patch_sizes = [profile[1] for profile in profiles]
      if centers is None:
        centers, labels = generate_pos_neg_label_crop_centers(  patch_sizes, 
                                                        num_samples=sample_size, 
                                                        pos_ratio=pos_ratio, 
                                                        label_spatial_shape=image.shape[:2],
                                                        fg_indices=fg_indices, 
                                                        bg_indices=bg_indices, 
                                                        rand_state=gnt, 
                                                        allow_smaller=False)
                                                        
    patches = [image[center[0] - patch_size[0] // 2:center[0] + patch_size[0] // 2, 
                      center[1] - patch_size[1] // 2:center[1] + patch_size[1] // 2].copy() 
                      for center, patch_size in zip(centers, patch_sizes)]
    output_sizes =  [profile[2] for profile in profiles]
    # print(np.where(np.array(patch_sizes) == 0)[0])
    # print(np.where(np.array([patch.shape for patch in patches]) == 0)[0])
    inputs = iter([(patch, transform, output_size, [header['h'], header['w']]) for patch, transform, output_size in zip(patches, [profile[0] for profile in profiles], output_sizes)])

    with multiprocessing.Pool(processes=num_workers) as p:
        outputs = p.starmap(warp_transform, tqdm.tqdm(inputs, total=sample_size, desc="Augmentation processing", leave=False))
    
    return outputs, profiles, centers, labels

if __name__ == "__main__":
    data = cv2.imread("data/v2/2015.png", cv2.IMREAD_UNCHANGED)
    print(data.shape)
    image, mask = data[:, :, :3], data[:, :, 3]
    gnt = np.random.RandomState(12)
    [bg_indices, fg_indices] = map_classes_to_indices(mask, [0, 1], max_samples_per_class=None)
    print(len(bg_indices), len(fg_indices))
    sample_size = 256
    aug = {'shear_range': 0.5, 'rotate_range': np.pi, 'scale_range': 0.3, 'flip_ratio': 0.5}
    header = {'h': 256, 'w':256}
    patches, profiles, centers = cut_centers_transform(data, gnt, fg_indices, bg_indices, header, aug, profiles=None, centers=None, pos_ratio=0.4)
    patches, _, _ = cut_centers_transform(data, gnt, fg_indices, bg_indices, header, None, profiles=profiles, centers=centers)
    images = [patch[:, :, :3] for patch in patches]
    images = [255 * patch[:][:, :, [3]] + image * ( 1 - patch[:, :, [3]]) for image, patch in zip(images, patches)]
    output = "data/viz"
    for idx, image in enumerate(images):
      print(image.shape)
      cv2.imwrite(f"data/viz/{idx}.png", image)
