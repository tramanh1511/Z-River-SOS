from typing import Any, Dict, List, Optional, Tuple
import hydra
from  omegaconf import DictConfig, OmegaConf
import pandas as pd
import os
import sys
from copy import copy
import shutil
import logging
logging.disable(logging.CRITICAL + 1) 
os.environ['HYDRA_FULL_ERROR'] = '1'
print(sys.getrecursionlimit())
sys.setrecursionlimit(1000000000)

import tqdm
import json
import rootutils
from pathlib import Path
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.zero_shot import *

from collections import defaultdict
from functools import  partial
from queue import PriorityQueue, Queue
import warnings
import heapq
from heapq import heappush, heappop


import cv2
import numpy as np
from PIL import  Image
import scipy
import skimage

import segmentationmetrics as segmetrics

import torch
print(torch.cuda.is_available())
import sam2
from sam2.build_sam import build_sam2, build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor as SAM

from hydra.core.global_hydra import GlobalHydra

import matplotlib.pyplot as plt
import matplotlib as mpl
from src.zero_shot.sam import SamInferer, roc_curve
from src.zero_shot.utils import save_gray, save_grey, sigmoid, plot_metric_distributions
import src.zero_shot.utils as utils

if GlobalHydra.instance().is_initialized():
    print("GlobalHydra is already initialized. Reinitializing")
else:
    print("GlobalHydra is not initialized.")

print("IMPORTED")

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def prepare(out_dir):
    # os.system(f"rm -rf {out_dir}")
    # if os.path.isdir(out_dir):
    #     shutil.rmtree(out_dir)
    (out_dir / "strong").mkdir(parents=True, exist_ok=True)
    (out_dir / "weak").mkdir(parents=True, exist_ok=True)
    (out_dir / "logit").mkdir(parents=True, exist_ok=True)
    (out_dir / "output").mkdir(parents=True, exist_ok=True)
    (out_dir / "image").mkdir(parents=True, exist_ok=True)
    (out_dir / "thin").mkdir(parents=True, exist_ok=True)
    (out_dir / "skeleton").mkdir(parents=True, exist_ok=True)
    (out_dir / "ensembled_output").mkdir(parents=True, exist_ok=True)
    (out_dir / "confidence").mkdir(parents=True, exist_ok=True)

def prepare_inspect(inspect_dir):
    (inspect_dir / "logit").mkdir(parents=True, exist_ok=True)
    (inspect_dir / "mask").mkdir(parents=True, exist_ok=True)
    (inspect_dir / "weight").mkdir(parents=True, exist_ok=True)

cmap = mpl.colormaps['viridis']

@hydra.main(version_base="1.3", config_path="../configs", config_name="sam.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    skip = -1
    data_dir = Path(cfg.paths.data_dir)
    # data_dir.mkdir(parents=True, exist_ok=True)
    # print(data_dir)'
    marker = cv2.imread(r"data\negative_region.png")[..., -1]
    marker = np.ones_like(marker, dtype=np.uint8)
    negatives = np.array([[275, 1100], [365, 1360], [900, 2050], [910, 2050], [5964, 3275]])
    
    builder = hydra.utils.instantiate(cfg.model.predictor)
    configs = OmegaConf.to_container(cfg.model.config)
    print(configs.keys())
    inspect = np.array(cfg.inspect)[:, ::-1].copy()
    manual = ["2015.png"]
    # print(np.unique(marker, return_counts=True), marker.shape)
    assert os.path.isdir(data_dir), "Data dir must be a dir"
    for index, file in enumerate(manual):
        # print(file)
        year = file.split(".")[0]
        kwargs = configs[f"y{year}"]
        print(kwargs)
        param = builder(**kwargs)
        param.label_bins = np.linspace(0, 0.9, 51).tolist() + np.linspace(0.9, 1., 51).tolist()
        image_path = os.path.join(data_dir, file)
        root_dir = Path(cfg.paths.output_dir)
        # plt.imshow(skimage.morphology.skeletonize(output['possible']))
        # print("Inferencing on SAMPLE", image_path)
        # Prepare for ablation study
        for beta in [param.beta]:
            # print("Inferencing on beta", beta)
            param.beta = beta
            trial = 0
            for trial in range(5):
                if skip > 0:
                    skip -= 1
                    continue
                param.reset()
                out_dir = root_dir /  f"{year}" / f"{trial:02d}"
                prepare(out_dir)
                eval_res = {'dice': [], 'iou': [], 'acc': [], 'recall': [], 'f1': []}
                rc_curves = []
                label_prob_map = []
                positives = []
                iterations = 0
                param.read(image_path)
                
                param.add_queue({'pt': np.array([1670, 2100])}, prior=2, isroot=True)
                param.add_queue({'pt': np.array(copy(OmegaConf.to_container(cfg.prompt)))}, isroot=True)
                param.neg = np.array([[275, 1100], [365, 1360], [900, 2050], [910, 2050], [5964, 3275]])
                param.marker = marker
                for pt in negatives:
                    cv2.circle(param.marker, pt[::-1], radius=30, color=0, thickness=-1)
                pbar = tqdm.tqdm(range(300), desc="Inference process")
                for iter in pbar:
                    pbar.refresh()
                    pbar.set_description_str(f"{year}_{beta:.2f}_{trial:02d}")
                    filename = f"iter_{iter}"
                    if len(param.queue) == 0:
                        break
                    iterations += 1
                    # print(f"Iteration {iterations}")
                    output = param.iter(debug=True, inspect=inspect.copy())
                    if output['ret'] is True:      
                        inspect_res = output['inspect']
                        inspect[..., 0] += inspect_res['add']
                        profiles = output['inspect']['profile']
                        for profile in profiles:
                            if not profile['ret']:
                                continue
                            ins_folder = out_dir / "inspect" / str(profile['index'])
                            if not os.path.isdir(ins_folder):
                                prepare_inspect(ins_folder)
                            save_gray(f"{ins_folder}/logit/iter_{profile['iter']}.jpg", sigmoid(profile['logit'] + 0.01) if not param.post_act else profile['logit'], 'viridis', output_size=cfg.log_size)
                            save_gray(f"{ins_folder}/mask/iter_{profile['iter']}.jpg", profile['mask'],  'viridis', nonzero=False,output_size=cfg.log_size)
                            save_gray(f"{ins_folder}/weight/iter_{profile['iter']}.jpg", profile['weight'], 'viridis', nonzero=False, output_size=cfg.log_size)     
                        if eval_res is None: 
                            eval_res = dict(zip(output['metrics'].keys(), [[]] * len(output['metrics'])))
                        # print(output['metrics'])
                        # print(output['infer']['logit'].min())
                        save_gray(f"{out_dir}/logit/{filename}.jpg", sigmoid(np.concatenate([sigmoid(output['infer']['logit']), output['prob_map']], axis=0)), 'viridis', output_size=cfg.log_size)
                        save_gray(f"{out_dir}/thin/{filename}.jpg", output['thin'], 'viridis', nonzero=False, output_size=cfg.log_size)
                        save_gray(f"{out_dir}/weak/{filename}.jpg", output['possible'], 'viridis', nonzero=False, output_size=cfg.log_size)
                        save_gray(f"{out_dir}/strong/{filename}.jpg", output['beta'], 'viridis', nonzero=False, output_size=cfg.log_size)
                        # save_gray(f"{out_dir}/output/{filename}.jpg", output['prob_map'], 'viridis', output_size=cfg.log_size)
                        # save_gray(f"{out_dir}/confidence/{filename}.jpg", np.concatenateoutput['prob_map'], 'viridis', output_size=cfg.log_size)
                        save_gray(f"{out_dir}/skeleton/{filename}.jpg", output['skeleton'], 'viridis', invert=True, output_size=cfg.log_size)
                        image = output['infer']['input'].copy().astype('float') / 255
                        input_mask = scipy.ndimage.morphological_gradient(output['infer']['inp_mask'], size=3)
                        src, dst = param.root, param.root + param.patch_size
                        label_mask = scipy.ndimage.morphological_gradient(param.label[src[0]:dst[0], src[1]:dst[1]].copy(), size=3)
                        pred_mask = scipy.ndimage.morphological_gradient(output['beta'].copy(), size=3)
                        # # print(image.shape, input_mask.shape, label_mask.shape, pred_mask.shape)
                        image = image * (1 - np.stack([input_mask, label_mask, pred_mask], axis=-1).max(axis=-1)[..., None]) \
                                + input_mask[..., None] * np.array([1, 0, 0])[None, None, :] \
                                + label_mask[..., None] * np.array([0, 1, 0])[None, None, :] \
                                + pred_mask[..., None] * np.array([0, 0, 1])[None, None, :]
                        annotation = output['infer']['pts']
                        positives.append(annotation[output['infer']['label'] > 0] + src)
                        a_label = output['infer']['label'][:, None]
                        color = [0, 1, 0, 0.5] * a_label + [1, 0, 0, 0.5] * (1 - a_label)
                        for pt, c in zip(annotation, color):
                            cv2.circle(image, pt, 5, c, -1)
                        cv2.imwrite(f"{out_dir}/image/{filename}.jpg", (image * 255)[..., ::-1].astype(np.uint8))
                        for key in output['metrics'].keys():
                                eval_res[key].append(float(output['metrics'][key]))
                        branches = [np.array(branch) for (cost, branch) in output['branches']]
                        for i, branch in enumerate(branches):
                            c_val = list(cmap(0.1 + 0.9 * (float(i) / len(branches))))
                            c_val[-1] = 0.3
                            # rgb = (int(c_val[0] * 255), int(c_val[1] * 255), int(c_val[2] * 255))
                            cv2.polylines(image, [branch[:, ::-1]], False, c_val, 1)
                            cv2.circle(image, branch[0, ::-1], 10, c_val[:3], 1)
                        if label_mask.mean() > 0.004: 
                            rc_curves.append(output['rc_curve'])
                            label_prob_map.append(output['label_prob_map'])
                    else:
                        # print("Hehe")
                        pass
                print(param.roi)
                with open(f"{out_dir}/micro_{'_'.join([str(item) for item in param.roi[2:]])}.txt", "w") as file:
                    file.write(f"Inferences: {iter}\n")
                    for key in eval_res.keys():
                        mean = np.mean(eval_res[key])
                        std = np.std(eval_res[key])
                        file.write(f"{key} metrics: {mean} +- {std}\n")
                df = pd.DataFrame(eval_res)
                df.to_csv(f"{out_dir}/micro_{'_'.join([str(item) for item in param.roi[2:]])}.csv")
                src, dst = [0, 0], param.box
                # src[0] = max(400, src[0])
                # weight = scipy.ndimage.gaussian_filter(param.weight, sigma=2.)
                for curve in rc_curves:
                    plt.plot(curve[..., 0], curve[..., 1], c=[0, 0, 1, 0.1])
                
                plt.plot([0, 1], [0, 1], color="orange", linestyle='--', label="Random Predictor")
                plt.title("ROC curves of all iterations")
                plt.savefig(f"{out_dir}/micro_roc.jpg")
                plt.close()
                plt.pause(0.2)
                np.save(f"{out_dir}/micro_roc_curves.npy", np.stack(rc_curves, axis=0))
                label_prob_hist = np.stack(label_prob_map, axis=0).sum(axis=0)
                label_prob_hist /= label_prob_hist.sum()
                np.save(f"{out_dir}/label_prob.npy", label_prob_hist)
                plt.plot(param.label_bins[:-1], label_prob_hist)
                plt.title("Model's confidence histogram on true labels")
                plt.savefig(f"{out_dir}/label_prob.jpg")
                plt.close()
                plot_metric_distributions(eval_res, prefix="Micro performance statistics", out_dir=out_dir, show=False)
                logit = param.logits[src[0]:dst[0], src[1]:dst[1]] / param.weight
                if not param.post_act:
                    prob = sigmoid(logit)
                else:
                    prob = logit
                sv = [0, 
                    param.alpha / 2, 
                    np.quantile(prob[prob >= param.alpha], 0.2), 
                    np.quantile(prob[prob >= param.alpha], 0.5), 
                    1]
                dv = [0, 0, param.alpha * 0.8 + 0.2, param.alpha * 0.5 + 0.5, 1]
                prob = scipy.ndimage.gaussian_filter(np.interp(prob, sv, dv), sigma=0.5)
                # beta = param.beta / param.weight
                # logit = logit ** 2
                label = param.label[src[0]:dst[0], src[1]:dst[1]]
                save_grey(f"{out_dir}/ensembled_output/logit.jpg", prob)
                save_grey(f"{out_dir}/ensembled_output/mask.jpg", param.b_mask)
                quantized = np.interp(prob ** 1.5, (0, 1), (0, 255)).astype(int).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(251, 251))
                quantized = clahe.apply(quantized)
                # print(quantized.shape, quantized.dtype)
                save_grey(f"{out_dir}/ensembled_output/quantized.jpg", quantized)
                b_mask = (quantized >= 150)
                b1 = cv2.adaptiveThreshold(quantized, 
                                                    1, 
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY, 
                                                    91, 
                                                    -15) * (quantized > 50) 
                b2 = cv2.adaptiveThreshold(quantized, 
                                                    1, 
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY, 
                                                    121, 
                                                    -15) * (quantized > 80)
                mask = b_mask + (1 - b_mask) * np.maximum(b1, b2)
                mask = mask + (1 - mask) * param.b_mask[src[0]:dst[0], src[1]:dst[1]] * (quantized >= 10)
                # mask *= param.marker
                mask = utils.prune(mask, min_size=100)
                output = np.zeros(mask.shape, np.uint8)
                _, label_im = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
                labs, counts = np.unique(label_im, return_counts=True)
                for lab, count in zip(labs[1:], counts[1:]):
                    im = label_im == lab
                    if param.b_mask[src[0]:dst[0], src[1]:dst[1]][im].sum() <= 5:
                        continue 
                    else: 
                        output = np.maximum(output, im)
                mask = output
                mask = utils.prune(mask, min_size=500)
                macro_roc = roc_curve(quantized.astype(float) / 255, label, step=20)
                np.save(f"{out_dir}/macro_roc_curve.npy", macro_roc)
                mask_grad = scipy.ndimage.morphological_gradient(mask, size=3)
                label_grad = scipy.ndimage.morphological_gradient(label, size=3)
                image = param.image.copy()
                res = segmetrics.SegmentationMetrics(mask,label,(1, 1))
                image[...,0][mask_grad > 0] = 255
                image[...,1][label_grad > 0] = 255
                positives = np.concatenate(positives, axis=0)
                for pt in positives:
                            cv2.circle(image, pt[::-1], 5, [255, 255, 0], -1)
                with open(f"{out_dir}/macro_{beta}.txt", "w") as file:
                    file.write(str(res.get_df()))
                df = res.get_df()
                df.to_csv(f"{out_dir}/macro_{beta}.csv", "w")
                cv2.imwrite(f"{out_dir}/ensembled_output/annotated.jpg", image[..., ::-1])
                pbar.write(f"{year}_{beta:.2f}_{trial:02d}:\n" + res.get_df().to_string())
    return True

if __name__ == "__main__":
    print("Proceed")
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    main()
