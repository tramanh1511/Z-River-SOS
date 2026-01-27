from collections import defaultdict
from functools import  partial
from queue import PriorityQueue, Queue
import warnings
import heapq
from heapq import heappush, heappop

import torch
import sam2
from sam2.build_sam import build_sam2, build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor as SAM
import matplotlib.pyplot as plt

import cv2
import numpy as np
from PIL import  Image
import scipy
import skimage
import src.zero_shot.utils as utils

MODEL_NAME = {
    "tiny": "sam2.1_hiera_tiny",
    "small": "sam2.1_hiera_small",
    "base": "sam2.1_hiera_base_plus",
    "large": "sam2.1_hiera_large"
}

clip = lambda x, low=0, high=1: max(low, min(high, x))
crop = lambda image, centroid, patch_size: image[centroid[0] - patch_size[0] // 2: centroid + patch_size[0] // 2, 
                                                    centroid[1] - patch_size[1] // 2: centroid + patch_size[1] // 2]
def roc(pred, label):
    tpr = (pred * label).sum() / label.sum()
    fpr = (pred * ( 1- label)).sum() / (1- label).sum()
    return [fpr, tpr]
 
def roc_curve(logit, label, step=20): 
    curve = np.array([roc(logit >= threshold, label) for threshold in np.linspace(0, 1, num=step)])
    curve = curve[curve[:, 0].argsort()]
    return curve

def eval(pred, label):
    false_region = cv2.dilate(label, np.ones([3, 3], dtype=np.uint8), iterations=10) - label 
    total = label.shape[0] * label.shape[1]
    intersection = pred * label
    int_count = intersection.sum()
    union = cv2.bitwise_or(pred, label)
    u_count = union.sum()
    dice = 2 * int_count / (int_count + u_count + 1)
    iou = int_count / (u_count + 1) 
    acc = (pred == label).sum() / total
    recall = int_count / (pred.sum() + 1)
    f1 = 2 * acc * recall / (acc + recall)  
    return {'dice': dice,
            'iou': iou,
            'acc': acc,
            'recall': recall,
            'f1': f1}

# A wrapper for Sam inference
class SamInferer:
    def __init__(self, cfg = "", 
                    ckpt: str = "",
                    model_id: str | None = None,
                    patch_size = [256, 256],
                    roi=[256, 256],
                    root_area=500,
                    max_roots=3,
                    patience = 5,
                    back_off = 5,
                    alpha=0.1, 
                    d_alpha=0.2,
                    beta=0.5, 
                    post_act=True, 
                    min_length=5,
                    kernel_size=3,
                    fill_kernel_size=5,
                    neg_dis=15,
                    pos_dis=15,
                    pos_rad=256,
                    pos_sc=1.,
                    sampling_dis=3,
                    neg_sampling_grid=6,
                    thresh=0.75,
                    decay=0.5,
                    confidence=0.8,
                    topk=2,
                    stable_weight=3,
                    gamma=1,
                    label_bins=None
                    ):
        if model_id is None:
            self.predictor = SAM(build_sam2(cfg, ckpt, device="cpu"))
        else: 
            self.predictor = SAM(build_sam2_hf(model_id))
        # Guidance and queries
        self.sampling_dis = sampling_dis
        self.marker = None
        self.hist = None
        self.patience = patience
        self.stable_weight = stable_weight
        self.back_off = back_off
        # self.queue = PriorityQueue()
        self.queue = []

        # Positive sampling
        self.pos = np.zeros([0, 2], dtype=int)
        self.pos_dis = pos_dis
        self.pos_rad = pos_rad
        self.pos_sc = pos_sc
        # Negative sampling
        self.neg = np.zeros([0, 2], dtype=int)
        self.neg_dis = neg_dis
        self.neg_sampling_grid = neg_sampling_grid
        
        # We gonna prioritize long flow over short ones
        self.root = None

        # Context related
        self.step = 0
        self.a_mask = None
        self.b_mask = None
        self.image = None 
        self.label = None
        self.alpha = alpha
        self.d_alpha = d_alpha
        self.beta = float(beta)
        self.confidence = confidence
        self.decay = decay
        self.post_act = post_act
        self.weight = None
        self.logits = None # Post-sigmoid or pre-sigmoid dependent
        self.var = None
        
        # kernel configuration
        self.patch_size = np.array(patch_size)
        self.w_kernel = [cv2.getGaussianKernel(patch_size[0], roi[0]), cv2.getGaussianKernel(patch_size[1], roi[1])]
        self.w_kernel = self.w_kernel[0] * self.w_kernel[1].T
        self.w_kernel /= self.w_kernel.max() 
        # TO prevent vanishing
        # self.w_kernel = np.interp(self.w_kernel, np.quantile(self.w_kernel, (0, 1)), (.2, 1))
        self.gamma = gamma
        
        # Uncertainty modelling
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        fill_kernel_size = (fill_kernel_size, fill_kernel_size) if isinstance(fill_kernel_size, int) else fill_kernel_size
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        self.fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, fill_kernel_size)
        self.stable_thresh = thresh
        
        # Flow skeleton
        self.atlas = None
        self.traits = None
        self.dis_map = None
        self.graph = defaultdict(list)
        self.min_length = min_length
        self.parent = dict()
        self.roi = None
        self.graph_root = np.zeros([0, 2], dtype=int)
        self.root_area = root_area
        self.max_roots = max_roots
        self.topk = topk

        # For conformalization
        self.label_bins = np.linspace(0, 1, num=40) if label_bins is None else label_bins
    
    def reset(self):
        self.step = 0
        self.queue = []
        self.pos = np.zeros([0, 2], dtype=int)
        self.neg = np.zeros([0, 2], dtype=int)
        self.neg_seg = None
        
        self.a_mask = None
        self.b_mask = None
        self.image = None 
        self.label = None
        self.weight = None
        self.logits = None # Post-sigmoid or pre-sigmoid dependent
        self.var = None
        
        # Flow skeleton
        self.atlas = None
        self.traits = None
        self.dis_map = None
        self.graph = defaultdict(list)
        self.parent = dict()
        self.roi = None
        self.graph_root = np.zeros([0, 2], dtype=int)

    def predict(self, src, dst): 
        logit = self.logits[src[0]:dst[0], src[1]:dst[1]] / self.weight[src[0]:dst[0], src[1]:dst[1]]
        prob_map = utils.sigmoid(logit) if not self.post_act else logit
        quantized = np.interp(prob_map ** 1.5, (0, 1), (0, 255)).astype(np.uint8)
        beta_mask = (quantized >= int(self.beta * 255)).astype(np.uint8)
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
        mask = np.maximum(beta_mask, np.maximum(b1, b2))
        mask = utils.prune(mask, min_size=500)
        return mask

    def pop(self):
        if len(self.queue) == 0:
            # print("Empty queue !!!")
            return None
        (score, item) = self.queue.pop(0)
        # print(f"Candidate of score {score}")
        return item
            
    def valid_pts(self, pts):
        pts -= self.root[None, :]
        check = np.ones(pts.shape[0])
        for i in range(pts.shape[1]):
            check *= np.where((pts[:, i] > 0) & (pts[:, i] < self.patch_size[i]), 1, 0)
        return pts[np.where(check > 0)]  
        
    def compose_prompts(self, pos, valid_neg):
        valid_pos = self.valid_pts(pos.copy())
        pos_label = np.ones(valid_pos.shape[0])
        # Generated locally so no need for projection
            
        if valid_neg.shape[0] == 0:
            return valid_pos[:, ::-1].copy(), pos_label
        neg_label = np.zeros(valid_neg.shape[0])
        # Must be in xy format
        return np.concatenate([valid_pos, valid_neg], axis=0)[:, ::-1].copy(), np.concatenate([pos_label, neg_label], axis=0)

    def read(self, image_path, channels=3):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if img.shape[-1] > channels: 
            self.label = img[..., channels]
        # Sharpen for better sense of boundary
        # hsv_image = cv2.cvtColor(img[..., :channels], cv2.COLOR_BGR2HSV_FULL)
        # unsharp = cv2.GaussianBlur(hsv_image, (3, 3), 0)
        # hsv_image = 2 * hsv_image - unsharp
        # self.image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB_FULL)
        self.image = img[..., :channels][..., ::-1]
        
        self.a_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.b_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.hist = np.zeros(self.image.shape[:2])
        self.weight = np.full(self.image.shape[:2], 1e-6)
        self.logits = np.zeros(self.image.shape[:2]) if self.post_act else np.full(self.image.shape[:2], -10.) * self.weight
        self.var = np.zeros_like(self.logits)
        # self.beta = np.full(self.image.shape[:2], self.beta) * self.weight
        self.output = np.zeros_like(self.a_mask)
        self.box = self.image.shape[:2]

        self.atlas = np.zeros_like(self.b_mask)
        self.traits = np.zeros_like(self.b_mask)
        self.dis_map = np.zeros([img.shape[0], img.shape[1], 2], dtype=float)
    
    # Generate negative prior
    def negative_sampling(self, pos, debug = False):
        # Prepare
        pts = self.valid_pts(pos.copy())
        dst = self.root + self.patch_size
        a_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.neg_dis, self.neg_dis))
        grad_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        
        mask = self.b_mask[self.root[0]:dst[0], self.root[1]:dst[1]].copy()        
        gradient = cv2.morphologyEx(cv2.dilate(mask, a_kernel, iterations=5), cv2.MORPH_GRADIENT, grad_kernel)

        for pt in pts:
            cv2.circle(gradient, pt[::-1], self.neg_dis * 3, 0, -1)

        alpha_mask = scipy.ndimage.binary_fill_holes(self.a_mask[self.root[0]: dst[0], self.root[1]:dst[1]]).astype(int).astype(np.uint8)
        alpha_mask = cv2.dilate(alpha_mask,  a_kernel, 3)

        negative_field = gradient * (1 - alpha_mask)
        # Discretize to 5 bin
        output = utils.grid_sampling(negative_field.astype(float), grid=self.neg_sampling_grid, alpha=0.01)
        if self.neg.shape[0] > 0:
            negative = self.valid_pts(self.neg)
            output = np.concatenate([negative, output])
        dis = np.triu(np.linalg.norm(output[None, :] - output[:, None, :], axis=2))
        dis[dis == 0] = 1e5
        drop = np.where(dis < self.neg_dis)[0]
        accepted_neg = [i for i in range(output.shape[0]) if i not in drop]
        output = output[accepted_neg]
        
        return {'pts': output} if not debug else {'pts': output, 'field': negative_field, 'b': alpha_mask, 'a': gradient}


    def roi_(self, x, y):
        if self.roi is None:
            self.roi = np.array([y, x, y + self.patch_size[0], x + self.patch_size[1]])
            return 
        self.roi[2] = max(self.roi[2], y + self.patch_size[0])
        self.roi[0] = min(self.roi[0], y)
        self.roi[3] = max(self.roi[3], x + self.patch_size[1])
        self.roi[1] = min(self.roi[1], x)

    def get_inbound(self, src, dst):
        dis_map = self.dis_map[src[0]:dst[0], src[1]:dst[1]]
        ske_map = np.linalg.norm(dis_map, axis=2)
        mask = 1 - scipy.ndimage.binary_erosion(np.ones_like(ske_map), iterations=5)
        pos = np.stack(np.where(mask * ske_map > 0)).T.astype(int)
        if pos.shape[0] == 0:
            return {'point': np.zeros([0, 2], dtype=int),
                    'direction': np.zeros([0, 2], dtype=int)}
        # direction = np.stack([dis_map[pt[0], pt[1]] for pt in pos], axis=0)
        direction = dis_map[pos[:, 0], pos[:, 1]]
        center_di = (dst - src) / 2 - pos
        center_di /= np.linalg.norm(center_di, axis=1)[:, None]
        # Get dominant axis
        axis = np.argmax(np.abs(center_di / self.patch_size), axis=1)
        # print(axis)
        # print(np.stack([pos, center_di, direction], axis=1))
        # Take dominant 
        valid = (np.take_along_axis(direction * center_di, axis[:, None], axis=1) >= 0).flatten()

        # print(f"Direction: {direction}, valid in {valid}")
        return {'point': pos[valid].copy().reshape(-1, 2),
                'direction': direction[valid].copy().reshape(-1, 2)}

    
    def gather_prompts(self, src):
        def filter(pts, min_dis=50):
            dis = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            valid = np.triu(np.ones_like(dis), k=1)
            dis[valid==0] = 1e5
            s = (dis < min_dis).sum(axis=0)
            # print(dis, s)
            return pts[s == 0].copy()
        res = []
        index = 0
        if len(self.queue) > 0:
            while index < len(self.queue):
                (_, prompt) = self.queue[index]
                item = prompt['pt']
                dis = np.abs(src - item)
                pos = item - self.root
                if (dis > self.pos_rad).any() or ((pos >= self.patch_size) | (pos < 0)).any() or self.hist[item[0], item[1]] < self.pos_sc: 
                    index += 1
                else:
                    res.append(self.queue.pop(index)[1]['pt'])
            res = [src.copy()] + res
            # print("Res:", res)
            res = filter(np.array(res), min_dis=self.pos_dis)
            # print(f"Gathered {res.shape[0] - 1} prompts in single batch")
            if len(res) == 1:
                return np.ones([0, 2])
            return res[1:self.topk]
        else:
            return np.ones([0, 2])
        
    def infer(self, debug=False):
        nested_prompt = self.pop()
        prompt = nested_prompt['pt'].copy()
        if prompt is None:
            return {'ret': False}
        if self.hist[prompt[0], prompt[1]] >= self.patience * 2:
            # print(f"Point {prompt} got inferred {self.hist[prompt[0], prompt[1]]} times, skipping")
            return {'ret': False}
        if self.marker is not None: 
            if self.marker[prompt[0], prompt[1]] == 0: 
                return {'ret': False}
            
        x, y, _, _ = utils.get_bbox(self.box, prompt, self.patch_size)
        self.roi_(x, y)
        # Change base for referencing and post-processing.
        self.root = np.array([y, x])
        dst = self.root + self.patch_size
        graph_root = self.graph_root - self.root
        base_root = self.get_inbound(self.root, dst)
        # print(base_root)
        # Add graph root if it's present in the image
        if (graph_root * (graph_root - self.patch_size)  <= 0).all(): 
            root_di = self.patch_size / 2 - graph_root
            root_di /= (np.linalg.norm(root_di) + 1e-4)
            # print("Prompt presents in the patch")
            base_root['point'] =  np.concatenate([base_root['point'], graph_root], axis=0)
            base_root['direction'] = np.concatenate([base_root['direction'], root_di], axis=0)
        # Image in RGB format
        patch = self.image[self.root[0]:dst[0], self.root[1]:dst[1]].copy()
        self.predictor.set_image(patch)
        # Positional Preparation
        # input_mask = self.b_mask[self.root[0]:dst[0], self.root[1]:dst[1]].copy()
        input_mask = cv2.erode(self.predict(self.root, dst), self.fill_kernel, iterations=2)
        # input_mask = 0
        if 'branch' in nested_prompt.keys():
            available = 1 - scipy.ndimage.binary_fill_holes(input_mask).astype(np.uint8)
            path_map = np.zeros_like(input_mask)
            pts = nested_prompt['branch'] - self.root
            valid = (pts[:, 0] >= 0) & (pts[:, 0] < self.patch_size[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < self.patch_size[1])
            pts = pts[valid, :]
            # print("Branch in the prompt", pts.shape)
            path_map[pts[:, 0], pts[:, 1]] = 1
            self.hist[self.root[0]: dst[0], self.root[1]:dst[1]] += cv2.dilate(path_map, self.fill_kernel, iterations=30) * self.w_kernel
            path_map *= available
            input_mask = np.maximum(input_mask, path_map)
            # self.atlas[self.root[0]: dst[0], self.root[1]:dst[1]] = np.maximum(self.atlas[self.root[0]: dst[0], self.root[1]:dst[1]], path_map)
        # input_mask = np.maximum(input_mask, cv2.dilate(self.atlas[self.root[0]: dst[0], self.root[1]:dst[1]], np.ones([3, 3], dtype=np.uint8)))
        input_mask = cv2.resize(input_mask.astype(np.uint8), (256, 256))
        # input_mask = cv2.resize(self.predict(self.root, dst), (256, 256))
        pos = np.concatenate([np.array(prompt)[None, :], self.gather_prompts(prompt)], axis=0).astype(int)
        # print("Prompts:", pos)
        base_root['point'] = np.concatenate([base_root['point'], pos - self.root[None, :]], axis=0)
        base_root['direction'] = np.concatenate([base_root['direction'], np.zeros([pos.shape[0], 2])], axis=0)
        neg = self.negative_sampling(pos, debug=debug)
        annotation, a_label = self.compose_prompts(pos, neg['pts'])
        # Mind that mask input must halve the size, for size matching
        # print(input_mask.shape, a_label.shape, annotation.shape)
        masks, scores, logits = self.predictor.predict(point_coords=annotation, 
                                                        point_labels=a_label, 
                                                        mask_input=input_mask[None, :] if input_mask.max() > 0 else None, 
                                                        multimask_output=False)
        # cv2.resize(logits[0], self.patch_size, interpolation=cv2.INTER_LINEAR)
        return {'ret': True,
                'graph_root': base_root,
                'input': patch,
                'mask': masks[0], 
                'score': scores[0], 
                'logit': cv2.resize(cv2.GaussianBlur(logits[0], (3, 3), 0), self.patch_size, interpolation=cv2.INTER_LINEAR), 
                'pts': annotation, 
                'label': a_label,
                'inp_mask': cv2.resize(input_mask, self.patch_size),
                'prompt': prompt,
                'negative': neg}
    
    # Allow pipeline injection
    def morphology_centers(self, segmentation, weight, minArea=2400, minW=3.):
        # From an unknown respected lad
        def get_center_of_mass(cnt):
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            return cy, cx
        label_map = skimage.measure.label(segmentation, connectivity=2)
        labels, counts = np.unique(label_map, return_counts=True)
        # print(counts)
        centers = []
        out_seg = np.zeros_like(segmentation)
        for label, count in zip(labels[1:int(min(len(labels), 1 + self.max_roots))], counts[1:int(min(len(labels), 1 + self.max_roots))]): 
            if count > minArea: 
                mask = (label_map == label).astype(np.uint8)
                # Force the isle to be stable
                if weight[mask == 1].mean() <= minW:
                    continue
                out_seg[mask > 0] = 1
                cnt = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
                centers.append((get_center_of_mass(cnt)))
        return {'centers': np.array(centers) if len(centers) > 0 else np.zeros([0, 2]), 
                'mask': out_seg}

    def graph_search(self, w_map, graph_root, mask, b_mask=None):
        # print("Searching")
        dst = self.root + self.patch_size
        w_pts = np.array(np.where((w_map > 0) * (w_map < 5))).T
        # print("Skeleton size:", w_pts.shape[0])
        if w_pts.shape[0] > 4000:
            # print("Mask dilated too much")
            return False, None, None
        
        dist = np.linalg.norm(w_pts - graph_root, axis=1)
        nn = w_pts[np.argmin(dist)]
        # print(graph_root, nn)
        # Flow porting
        # Update available mask
        # ref_mask = self.b_mask[root[0]: dst[0], root[1]:dst[1]].copy()
        
        tree = utils.unilateral_dfs_tree( w_map.copy(), 
                                    mask, 
                                    tuple(nn), 
                                    b_mask=b_mask,
                                    alpha=0.01, 
                                    thresh=self.stable_thresh,
                                    dis_map=self.dis_map[self.root[0]: dst[0], self.root[1]:dst[1]].copy(),
                                    context_size=15,
                                    tolerance=-0.5
                                  )
        # unilateral_dfs_tree(g_mask, inp_mask, start, weight = 1, alpha=0.1, thresh=0.95, context_size=3, dis_map=None)
        # Ordering from leaves to roots
        self.dis_map[self.root[0]: dst[0], self.root[1]:dst[1]] = (self.dis_map[self.root[0]: dst[0], self.root[1]:dst[1]] + scipy.ndimage.gaussian_filter(tree['dis_map'], [5., 5., 0])) / 2
        # print("Maximal outtro and intro:", tree['status'].max())
        # self.traits[self.root[0]: dst[0], self.root[1]:dst[1]] = np.maximum(self.traits[self.root[0]: dst[0], self.root[1]:dst[1]], tree['status'] >= max(0, tree['status'].max() - 4))
        cost_map = tree['cost']
        branches = utils.longest_path_branching(tree['dfs_tree'], tuple(nn), dead=tree['dead'])
        valid_branches = [branches[i] for i in range(len(branches)) if len(branches[i]) >= self.min_length]
        # print(len(branches), len(valid_branches), [len(branch) for branch in branches])
        # Cost as normalized energy function along the geodesics
        costs = [cost_map[branch[0][0], branch[0][1]] / len(branch) for branch in valid_branches]
        return True, sorted(zip(costs, valid_branches), key=lambda x: x[0]), tree
    
    def iter(self, seg_res=None, debug=False, inspect=None):
        self.step += 1
        if seg_res is None:
            seg_res = self.infer(debug=debug) 
        if seg_res['ret'] is False:
            return seg_res
        dst = self.root + self.patch_size
        # Updating primitives
        # print(self.patch_size, seg_res['mask'].shape, seg_res['mask'].dtype)
        # Never let it lower than alpha
        score = max(seg_res['score'], self.alpha) 
        if score < self.alpha:
            warnings.warn(f"Model confidence {score} is hazardous, please make prompt to escape uncertainty")
        # print(f"Score {score}")
        # print(f"With {seg_res['label'].sum()} positives and {seg_res['label'].shape[0] - seg_res['label'].sum()} negatives")
        # Clean background from static gain
        prob_map = utils.sigmoid(seg_res['logit'])
        # Model was trained to get threshold at 0.5 intuitively
        if (prob_map >= self.alpha).sum() < self.root_area: 
            return {'ret': False}
        sv = [0, 
              self.alpha / 2, 
              np.quantile(prob_map[prob_map >= self.alpha], 0.2), 
              np.quantile(prob_map[prob_map >= self.alpha], 0.5), 
              1]
        dv = [0, 0, self.alpha * 0.8 + 0.2, self.alpha * 0.5 + 0.5, 1]
        # sv, dv = [0, 1], [0, 1]
        # print(sv, dv)
        if self.post_act:
            normed_map = scipy.ndimage.gaussian_filter(np.interp(prob_map, sv, dv), sigma=0.5) 
            # normed_map = prob_map
        else:
            normed_map = np.where(np.abs(seg_res['logit']) > 10, np.sign(seg_res['logit']) * 10, seg_res['logit'])
        normed_prob_map = normed_map if self.post_act is True else utils.sigmoid(normed_map)
        # print(f"Quantile range: {sv}")
        # Weight ensemble
        weight = score * self.w_kernel
        # When weight is low, skip it to the next pred
        ensemble_kernel = self.w_kernel / (self.weight[self.root[0]: dst[0], self.root[1]:dst[1]] + self.w_kernel)
        # Well, there is a case where confidence score is too low, so add tolerance 
        prev = self.logits[self.root[0]: dst[0], self.root[1]:dst[1]] / self.weight[self.root[0]: dst[0], self.root[1]:dst[1]]
        prev_prob = prev if self.post_act is True else np.interp(utils.sigmoid(prev), sv, dv)
        # prev_beta = self.beta[self.root[0]: dst[0], self.root[1]:dst[1]] / self.weight[self.root[0]: dst[0], self.root[1]:dst[1]]
        prev_var = self.var[self.root[0]: dst[0], self.root[1]:dst[1]].copy()
        # Update variance with momentum
        if (normed_prob_map > self.beta).sum() <= self.root_area:
            # print("Segmentation mask is self contained, skipping")
            return {'ret': False, 'infer': seg_res, 'logit': normed_map, 'prob': normed_prob_map}
        max_dev = 1 if self.post_act is True else min(5, max(2, -np.quantile(-seg_res['logit'], 0.0005) - np.quantile(seg_res['logit'], 0.05))) / 2
        self.var[self.root[0]: dst[0], self.root[1]:dst[1]] = (1 - ensemble_kernel) * self.var[self.root[0]: dst[0], self.root[1]:dst[1]] + ensemble_kernel * np.minimum((normed_map - prev) ** 2, ( 0.3 * max_dev ) ** 2)
        
        # Subtract to get gain properties
        true =  prev_prob > self.alpha
        positive = normed_prob_map > self.alpha
        # print(true.mean(), positive.mean())
        roi = skimage.morphology.binary_dilation(true | positive, footprint=np.ones([5, 5], dtype=np.uint8))
        dev = (self.var[self.root[0]: dst[0], self.root[1]:dst[1]] ** 0.5)[roi].mean()
        liou = ((true & positive).sum() + 1) / (true.sum() + 1)
        # print(dev.shape, beta.shape, prob_map.shape)
        offset = (normed_map - prev) / (dev + 1e-3)
        confidence = scipy.stats.norm.cdf(offset)
        # confidence = np.where(offset < 0, 1 - confidence, confidence)
        unstable = np.abs(confidence - 0.5) >= (self.confidence / 2) 
        stable = (prev_var < 0.05 * max_dev) & (self.weight[self.root[0]: dst[0], self.root[1]:dst[1]] >= self.confidence * self.stable_weight) & roi
        stable = stable | ((seg_res['inp_mask'] > 0) & positive) 
        # If the overlap is too shallow, preserves the ensemble output
        if liou < 0.5:
            tn = true & ~positive 
            fp = ~true & positive
            stable = stable | tn
            unstable = unstable & fp
        # Update properties
        self.logits[self.root[0]: dst[0], self.root[1]:dst[1]] += normed_map * weight
        self.logits[self.root[0]: dst[0], self.root[1]:dst[1]] += (-1 * stable + self.weight[self.root[0]: dst[0], self.root[1]:dst[1]] / weight * unstable)  * weight * self.decay * (normed_map - prev)  
        # Update decision boundary and weight
        # gain = np.sign(normed_prob_map - prev_beta) * (np.abs(confidence - 0.5) - self.confidence / 2) * dev 
        # self.beta[self.root[0]: dst[0], self.root[1]:dst[1]] += (prev_beta) * weight  
        self.weight[self.root[0]: dst[0], self.root[1]:dst[1]] += weight
        # Probabilistic Map
        logit_map = self.logits[self.root[0]: dst[0], self.root[1]:dst[1]] / self.weight[self.root[0]: dst[0], self.root[1]:dst[1]]
        prob_map = np.interp(utils.sigmoid(logit_map), sv, dv) if not self.post_act else logit_map
        
        # prob_map[unstable] = normed_prob_map[unstable]
        # self.logits[self.root[0]: dst[0], self.root[1]:dst[1]] = prob_map * self.weight[self.root[0]: dst[0], self.root[1]:dst[1]] * self.decay + (1 - self.decay) * self.logits[self.root[0]: dst[0], self.root[1]:dst[1]]
        # prob_map = prob_map ** 2
        # Discretized Probabilistic Map 
        quantized_conf = np.interp(prob_map, (0, 1), (0, 255)).astype(np.uint8)
        if not self.post_act: 
            prob_dev = np.abs(prob_map - prev_prob)[roi].mean()
        else:
            prob_dev = dev
        prob_map = prob_map ** self.gamma
        # print(f"Alpha adaptive threshold {- int(self.alpha * prob_dev * 255)}")
        # Discretized Probabilistic Map
        # beta = self.beta[self.root[0]: dst[0], self.root[1]:dst[1]] / self.weight[self.root[0]: dst[0], self.root[1]:dst[1]]
        # beta = (scipy.ndimage.gaussian_filter(beta, 2.) * 255).astype(np.uint8)
        beta_mask = (quantized_conf >= int(self.beta * 255)).astype(np.uint8)
        beta_mask = utils.prune(utils.morphology_closing(beta_mask, self.fill_kernel, 1), min_size=25)
        beta_mask[~unstable] = np.maximum(self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]][~unstable], beta_mask[~unstable])
        # self.b_mask[root[0]: dst[0], root[1]:dst[1]] = 
        # possible = (confidence >= self.alpha).astype(np.uint8)
        # possible -= possible * self.b_mask[root[0]: dst[0], root[1]:dst[1]]
        # possible = scipy.ndimage.binary_fill_holes(possible).astype(np.uint8)  * (1 - holes)
        # Get possible
        window_size = max(self.patch_size) // self.neg_sampling_grid
        window_size += window_size % 2 - 1
        discrete = cv2.adaptiveThreshold(quantized_conf, 
                                            1, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 
                                            window_size, 
                                            - max(15, int(self.confidence * prob_dev * 255)))
        beta_mask = beta_mask + (1 - beta_mask) * discrete
        holes = scipy.ndimage.binary_fill_holes(beta_mask) - beta_mask
        # Filter out soft edge from original beta mask
        possible = cv2.adaptiveThreshold(quantized_conf, 
                                            1, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 
                                            window_size, 
                                            - max(5, int(self.d_alpha * prob_dev * 255))) * (quantized_conf >= self.alpha * 255)
        possible = possible + (1 - possible) * beta_mask
        # Process 
        possible = utils.prune(possible, min_size=self.root_area)
        # possible = scipy.ndimage.binary_closing(possible, iterations=2).astype(np.uint8)
        possible = utils.smooth_mask(possible, 5, 1.5) * (1 - holes)
        
        if self.marker is not None: 
            marker = self.marker[self.root[0]: dst[0], self.root[1]:dst[1]]
            possible *= marker
            beta_mask *= marker
        else:
            marker = None
        # marker = None
        
        possible = np.maximum(possible, beta_mask)
        # possible = cv2.morphologyEx(possible, cv2.MORPH_CLOSE, self.fill_kernel, 2)
        
        # Stabilize skeleton
        thin, thin_w = utils.morphology_thinning(possible, return_weight=True)
        # print("Thin:", possible.shape, thin_w.shape, thin.shape, possible.max())
        thin_w = scipy.ndimage.grey_dilation(thin_w, size=3)
        stable = scipy.ndimage.grey_closing(thin_w + self.atlas[self.root[0]: dst[0],self.root[1]:dst[1]], 
                                            size=5)
        stable = utils.prune(stable, min_size=10) * stable
        
        skeleton = skimage.morphology.skeletonize(stable)
        stable = stable.astype(float)
        stable = stable / (stable[stable > 0].mean() + 1e-3)
        stable[stable == 0] = 1e2 
        # print("Stable:", stable.shape, "Skeleton values:", np.unique(skeleton))
        # skeleton = getLargestCC(skeleton).astype(int).astype(np.uint8)
        # Flow map generation output['dist']
        # print(score, skeleton.shape, prob_map.shape)
        w_map =  skeleton.astype(float) * ((1 - quantized_conf.astype(float) / 255) * (stable + 1e-1) + 1e-1) 
        
        # w_map /= w_map.max()
        w_map[self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]] * skeleton == 1] = np.maximum(w_map[self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]] * skeleton == 1], .1)
        w_map[w_map > 5] = 5 
        w_map /= 5
        w_map[~skeleton] = 1e5
        # print("Weight map",  w_map.max(), w_map.min())
        # save_gray("/work/hpc/potato/airc/data/vis/debug.jpg", w_map, 'viridis')
        # Update root to center of mass
        ref_mask = possible.copy()
        acp_mask = np.zeros_like(ref_mask)
        graph_root = seg_res['graph_root']
        # print(f"Graph roots: {graph_root['point']}")
        roots = []
        direction = []
        for i, root in enumerate(graph_root['point']):
            # print(root)
            root = np.array([max(0, min(sz, item)) for sz, item in zip(self.patch_size - 1, root)], dtype=int)
            if (acp_mask[root[0], root[1]] == 1) or (ref_mask[root[0], root[1]] == 0):
                continue
            roots.append(root)
            try:
                direction.append(graph_root['direction'][i]) 
            except:
                pass
                # print(f"Graph mismatch {len(graph_root['point'])} and {len(graph_root['direction'])}")
            # print(direction)
            temp = skimage.segmentation.flood(ref_mask, tuple(root.tolist()))
            # print(f"Component at {root} coverage: {temp.mean()}")
            # ref_mask[temp > 0] = 0
            acp_mask = np.maximum(acp_mask, temp)
        # Flag all inferences
        # self.hist[self.root[0]: dst[0], self.root[1]:dst[1]] += scipy.ndimage.binary_erosion(acp_mask, iterations=self.back_off - 1)
        remains = possible * ( 1 - acp_mask )
        if remains.max() > 0:
            cc_centers = self.morphology_centers(remains, 
                                                 self.weight[self.root[0]: dst[0], self.root[1]:dst[1]], 
                                                 minArea=self.root_area)
            # print("CC max value:", cc_centers['mask'].max())
        else:
            # print("Cluster is empty")
            cc_centers = {'centers': np.zeros([0, 2]), 'mask': np.zeros([0, 2])}
        roots = np.concatenate([np.array(roots).reshape(-1, 2), cc_centers['centers']], axis=0).astype(int)
        direction = np.pad(np.array(direction).reshape(-1, 2), ((0, cc_centers['centers'].shape[0]), (0, 0)), mode='constant')
        # import IPython; IPython.embed()
        # Given the fact that the prompting
        total_branches = []
        result_mask = np.zeros_like(ref_mask)
        canvas = np.zeros_like(thin_w)
        for root in roots:
            # print(root)
            # Update history
            if beta_mask[root[0], root[1]] == 1:
                result_mask[skimage.segmentation.flood(beta_mask, tuple(root.tolist())) > 0] = 1
            
            ret, valid_branches, tree = self.graph_search(w_map, 
                                               root, 
                                               result_mask,
                                                b_mask=beta_mask
                                              )
            if not ret: 
                # print("Tree not accepted")
                continue
            # valid_branches = [case[1] for case in valid_branches]
            cost = tree['cost']
            # state_map = scipy.ndimage.grey_dilation(tree['cost'], size=20)
            total_branches += valid_branches
            # valid_branches = sorted(valid_branches, key= lambda x: len(x))
            for branch_idx, (cost, branch) in enumerate(valid_branches[:self.topk]):
                # branch = smooth_graph(np.array(branch), 5, 1.5)
                branch = np.array(branch)
                i = 0
                (x, y) = self.patch_size / 2
                for i in range(self.back_off, len(branch)):
                    x, y = branch[i]
                    if confidence[x, y] > self.alpha:
                        # if state_map[x, y] == 1 or state_map[x, y] % 2 == 0:
                        break
                # print(branch[:10])
                for index, j in enumerate(range(i, max(i + 1, len(branch) - self.min_length), self.sampling_dis)):
                    x, y = branch[j]
                    score = prob_map[x, y] * (1000 - (len(branch) - j)) * cost * self.w_kernel[x, y]
                    # print(f"Candidate {j} or length {len(branch)}: ", self.root + branch[j], score)
                    self.add_queue(prompt={'pt':self.root + branch[j], 'branch': self.root + np.array(branch)}, prior = score)
                
                # Only get outgoing vertexes, 
                # if it loops into the main stream
                # Then its a hole and already solved by fill holes
                # if ref_mask[x, y] == 0: 
                #     # Roll back to prevent overflow
                # else: 
                #     print("Point in mask already")
                # Draw on canvas for mask extraction later
        # print(result_mask.max(), self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]].max(), beta_mask.max())
        self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]] = np.maximum(self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]], result_mask)
        # self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]][unstable] = result_mask[unstable].copy()
        self.atlas[self.root[0]: dst[0], self.root[1]:dst[1]] = np.maximum(self.atlas[self.root[0]: dst[0], self.root[1]:dst[1]], canvas)
        metrics = eval(self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]].copy(), self.label[self.root[0]: dst[0], self.root[1]:dst[1]])
        rc_curve = roc_curve(prob_map, self.label[self.root[0]: dst[0], self.root[1]:dst[1]].copy(), step=40)
        label_prob_map = np.histogram(prob_map[self.label[self.root[0]: dst[0], self.root[1]:dst[1]] > 0], bins=self.label_bins, density=True)[0]
        if debug:
            profiles = []
            update = None
            if inspect is not None:
                update = np.zeros(inspect.shape[0], dtype=int)
                for index, cpt in enumerate(inspect):
                    num, pt = cpt[0], cpt[1:]     
                    inp_src, inp_dst = pt - self.patch_size // 2, pt + self.patch_size // 2
                    if self.hist[inp_src[0]: inp_dst[0], inp_src[1]:inp_dst[1]].max() < num:
                        profiles.append({'ret': False})
                    else:
                        num += 1
                        update[index] = 1
                        profiles.append({   'index': index,
                                            'iter': int(num),
                                            'pt': pt,
                                            'ret': True, 
                                            'mask': self.b_mask[inp_src[0]: inp_dst[0], inp_src[1]:inp_dst[1]].copy(),
                                            'logit': self.logits[inp_src[0]: inp_dst[0], inp_src[1]:inp_dst[1]] / self.weight[inp_src[0]: inp_dst[0], inp_src[1]:inp_dst[1]],
                                            'weight': self.weight[inp_src[0]: inp_dst[0], inp_src[1]:inp_dst[1]]})  
            return {'ret': True,
                    'marker': marker,
                    'roots': {'pts': roots, 'directions': direction},
                    'infer': seg_res,
                    'confidence': confidence,
                    'beta': self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]].copy(),
                    'src_beta': beta_mask,
                    'prob_map': prob_map,
                    'stable': stable,
                    'thin': thin_w, 
                    'skeleton': w_map,
                    'branches': total_branches,
                    'possible': possible,
                    'canvas': canvas,
                    'metrics': metrics,
                    'rc_curve': rc_curve,
                    'label_prob_map': label_prob_map,
                    'inspect': {'profile': profiles, 'add': update}
                    }
        else: 
            return {'ret': True,
                    'infer': seg_res,
                    'branches': valid_branches,
                    'metrics': metrics}

    def update_graph(self, path):
        # Inverse sampling from root to leaves
        for i in range(len(path) - 1, 0, -1):
            self.bi_add(tuple(self.root + path[i-1]), tuple(self.root + path[i]))

    def bi_add(self, src, dst):
        self.graph[src].append(dst)
        self.graph[dst].append(src)

    def check_hist(self, pt):
        return self.hist[pt[0], pt[1]] > self.patience
    
    def add_queue(self, prompt, prior: float = 1, isroot: bool =False):
        if isroot:
            # self.pos = np.concatenate([self.pos, np.array([pt])], axis=0)
            # self.queue.put((prior, pt))
            heappush(self.queue, (prior, prompt))
            self.graph_root = np.array(prompt['pt'])[None, :]
            return 
        pt = prompt['pt']
        if self.check_hist(pt): 
            # print(f"Same position got infered for {self.hist[pt[0], pt[1]]} times, skipping {pt}")
            return
        score = max(1, int(prior + self.step))
        entry = (score, prompt)
        try:
            # self.queue.put(entry)
            self.queue.append(entry)
            self.queue = sorted(self.queue, key=lambda x: x[0])
            # print(self.queue)
        except Exception as error:
            print(error)
            pass
