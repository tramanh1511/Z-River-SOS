import torch
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor as SAM
from lang_sam.models.gdino import GDINO
from skimage import morphology as morph
import scipy
import skimage



import rootutils
rootutils.setup_root(__file__, indicator="setup.py", pythonpath=True)

from src.zero_shot.utils import *


crop = lambda image, centroid, patch_size: image[centroid[0] - patch_size[0] // 2: centroid + patch_size[0] // 2, 
                                                    centroid[1] - patch_size[1] // 2: centroid + patch_size[1] // 2]

def get_bbox(bound, pt, patch_size):
    x = min(bound[1] - patch_size[1], max(0, pt[1] - patch_size[1] // 2))
    y = min(bound[0] - patch_size[0], max(0, pt[0] - patch_size[0] // 2))
    return x, y, patch_size[0], patch_size[1]

# A wrapper for Sam inference
class LangSamInferer:
    def __init__(self, cfg = "", 
                    ckpt: str = "", 
                    patch_size = [512, 512], 
                    alpha=0.1, 
                    beta=0.5, 
                    post_act=True, 
                    min_length=5,
                    kernel_size=3,
                    fill_kernel_size=7,
                    neg_dis=15,
                    sampling_grid=8,
                    thresh=0.75):
        self.predictor = SAM(build_sam2(cfg, ckpt))
        self.gdino = GDINO()
        self.boxes = []
        # Guidance and queries
        self.pos = np.zeros([0, 2], dtype=int)
        self.queue = []

        # Negative sampling
        self.neg = np.zeros([0, 2], dtype=int)
        self.neg_dis = neg_dis
        self.sampling_grid = sampling_grid
        # We gonna prioritize long flow over short ones
        self.root = None

        # Context related
        self.a_mask = None
        self.b_mask = None
        self.image = None 
        self.alpha = alpha
        self.beta = beta
        self.post_act = post_act
        self.weight = None
        self.logits = None # Post-sigmoid or pre-sigmoid dependent
        
        # kernel configuration
        self.patch_size = np.array(patch_size)
        self.w_kernel = [cv2.getGaussianKernel(patch_size[0], patch_size[0] // 2), cv2.getGaussianKernel(patch_size[1], patch_size[1] // 2)]
        self.w_kernel = (self.w_kernel[0] / self.w_kernel[0][0, 0]) * (self.w_kernel[1] / self.w_kernel[1][0, 0]).T
        self.w_kernel /= self.w_kernel.sum() 
        # TO prevent vanishing
        self.w_kernel /= self.w_kernel.min()
        # Uncertainty modelling
        
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        fill_kernel_size = (fill_kernel_size, fill_kernel_size) if isinstance(fill_kernel_size, int) else fill_kernel_size
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        self.fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, fill_kernel_size)
        self.stable_thresh = thresh
        
        # Flow skeleton
        self.graph = defaultdict(list)
        self.min_length = min_length
        self.parent = dict()

        

    def compose_prompts(self):
        def valid_pts(pts):
            pts -= self.root[None, :]
            check = np.ones(pts.shape[0])
            for i in range(pts.shape[1]):
                check *= np.where((pts[:, i] > 0) & (pts[:, i] < self.patch_size[i]), 1, 0)
            return pts[np.where(check > 0)]
        
        valid_pos = valid_pts(self.pos.copy())
        pos_label = np.ones(valid_pos.shape[0])
        if self.neg.shape[0] == 0:
            return valid_pos, pos_label
        # Generated locally so no need for projection
        valid_neg = self.neg
        neg_label = np.zeros(valid_neg.shape[0])
        # Must be in xy format
        return np.concatenate([valid_pos, valid_neg], axis=0)[:, ::-1].copy(), np.concatenate([pos_label, neg_label], axis=0)

    def read(self, image_path, channels=3):
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[:, :, :channels]
        self.a_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.b_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.logits = np.zeros(self.image.shape[:2])
        self.weight = np.full(self.image.shape[:2], 1e-6)
        self.beta = np.full(self.image.shape[:2], self.beta * self.weight)
        self.output = np.zeros_like(self.a_mask)
        self.box = self.image.shape[:2]
    
    # Generate negative prior
    def negative_sampling(self, debug = False):
        # Prepare
        pts = (self.pos - self.root).round().astype(int)
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
        output = grid_sampling(negative_field.astype(float), grid=self.sampling_grid, alpha=0.01)

        return {'pts': output} if not debug else {'pts': output, 'field': negative_field, 'b': alpha_mask, 'a': gradient}


    def infer(self, debug=False):
        prompt = self.queue.pop(0)
        x, y, _, _ = get_bbox(self.box, prompt, self.patch_size)
        # Change base for referencing and post-processing.
        self.root = np.array([y, x])
        dst = self.root + self.patch_size
        # Image in RGB format
        patch = self.image[self.root[0]:dst[0], self.root[1]:dst[1], ::-1].copy()
        self.predictor.set_image(patch)

        neg = self.negative_sampling(debug=debug)
        self.neg = neg['pts']
        annotation, a_label = self.compose_prompts()
        # a_mask = cv2.resize(self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]], (256, 256))
        a_mask = self.logits[self.root[0]: dst[0], self.root[1]:dst[1]] / self.weight[self.root[0]: dst[0], self.root[1]:dst[1]]
        if not self.post_act:
            a_mask = sigmoid(a_mask)
        # Quantile for recall
        # Always lower bound it for measure
        score = max(0.4, np.quantile(a_mask, 0.98))
        a_mask = np.where(a_mask > score, 1, a_mask / score)
        
        a_mask = cv2.resize(a_mask, (256, 256), cv2.INTER_LINEAR)
        print(a_mask.max())
        # Mind that mask input must halve the size, for size matching
        print(a_mask.shape, a_label.shape, annotation.shape )
        masks, scores, logits = self.predictor.predict(point_coords=annotation, 
                                                        point_labels=a_label, 
                                                        mask_input=a_mask[None, :], 
                                                        multimask_output=False)

        return {'mask': masks[0], 
                'score': scores[0], 
                'logit': cv2.resize(logits[0], self.patch_size, interpolation=cv2.INTER_LINEAR), 
                'pts': annotation, 
                'label': a_label,
                'inp_mask': self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]].copy(),
                'prompt': prompt,
                'negative': neg}
    
    # Allow pipeline injection
    def iter(self, seg_res=None, debug=False):
        if seg_res is None:
            seg_res = self.infer(debug=debug) 
            
        dst = self.root + self.patch_size

        # Updating primitives
        print(self.patch_size, seg_res['mask'].shape, seg_res['mask'].dtype)
        score = seg_res['score'] 
        assert score > self.alpha, f"Model confidence {score} is hazardous, please make prompt to escape uncertainty"
        print(f"Score {score}")
        print(f"With {seg_res['label'].sum()} positives and {seg_res['label'].shape[0] - seg_res['label'].sum()} negatives")
        prob_map = sigmoid(seg_res['logit'])
        weight = score * self.w_kernel
        # Well, there is a case where confidence score is too low, so add tolerance 
        self.beta[self.root[0]: dst[0], self.root[1]:dst[1]] += weight * score
        if self.post_act:
            self.logits[self.root[0]: dst[0], self.root[1]:dst[1]] += prob_map * weight
        else: 
            self.logits[self.root[0]: dst[0], self.root[1]:dst[1]] += seg_res['logit'] * weight
        self.weight[self.root[0]: dst[0], self.root[1]:dst[1]] += weight
        
        # Subtract to get gain properties
        beta = self.beta[self.root[0]: dst[0], self.root[1]:dst[1]] / self.weight[self.root[0]: dst[0], self.root[1]:dst[1]]
        prob_map = self.logits[self.root[0]: dst[0], self.root[1]:dst[1]] / self.weight[self.root[0]: dst[0], self.root[1]:dst[1]]
        prob_map = sigmoid(prob_map) if not self.post_act else prob_map
        prob_map = prob_map ** 2
        self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]] = cv2.bitwise_or(self.b_mask[self.root[0]: dst[0], self.root[1]:dst[1]], (prob_map > beta).astype(np.uint8)) 
        possible = (prob_map >= self.alpha).astype(int)
        possible -= possible * seg_res['inp_mask'].astype(int)
        possible = possible.astype(np.uint8)
        self.a_mask[self.root[0]: dst[0], self.root[1]:dst[1]] = cv2.bitwise_or(self.a_mask[self.root[0]: dst[0], self.root[1]:dst[1]], possible)
        
        possible = cv2.morphologyEx(possible, cv2.MORPH_CLOSE, self.close_kernel, 1)
        possible = scipy.ndimage.binary_fill_holes(possible)
        possible = getLargestCC(possible).astype(int).astype(np.uint8)
        possible = smooth_mask(possible, 7, 3)
        # Stabilize skeleton
        thin, thin_w = morpholgy_thinning(possible, return_weight=True)
        stable = cv2.morphologyEx(thin_w, cv2.MORPH_DILATE + cv2.MORPH_CLOSE, self.fill_kernel, 3)
        stable = prune(stable, thin_w.max() * self.stable_thresh, min_size=25)
        # stable = smooth_mask(stable, 7, 3)
        # stable = getLargestCC(stable).astype(int).astype(np.uint8)
        # Skeletonize pruned mask
        skeleton = skimage.morphology.skeletonize(stable)
        # skeleton = getLargestCC(skeleton).astype(int).astype(np.uint8)
        # Flow map generation 
        w_map = (skeleton * prob_map) / score
        w_map[seg_res['mask'] * skeleton == 1] = 1
        w_map[w_map > 1] = 1 

        # Given the fact that the prompting 
        w_pts = np.array(np.where(w_map == 1)).T
        dist = np.linalg.norm(w_pts - (seg_res['prompt'] - self.root), axis=1)
        nn = w_pts[np.argmin(dist)]
        # Flow porting
        self.bi_add(tuple(seg_res['prompt']), tuple(self.root + nn))
        ref_mask = cv2.dilate(seg_res['inp_mask'], self.close_kernel, 2)
        tree = dfs_tree(w_map.copy(), 
                        ref_mask, 
                        tuple(nn), 
                        alpha=0.01, 
                        thresh=0.8)

        # Ordering from leaves to roots
        branches = longest_path_branching(tree['dfs_tree'], tuple(nn))
        valid_branches = [branches[i] for i in range(len(branches)) if len(branches[i]) >= self.min_length]
        for branch in valid_branches:
            x, y = branch[0]
            # Only get outgoing vertexes, 
            # if it loops into the main stream
            # Then its a hole and already solved by fill holes
            if ref_mask[x, y] == 0: 
                # Roll back 5 step to prevent overflow
                self.add_queue(self.root + branch[3])
                self.update_graph(branch)

        if debug:
            return {'infer': seg_res,
                    'stable': stable,
                    'thin': thin_w, 
                    'branches': valid_branches,
                    'possible': possible,
                    }
        else: 
            return {'infer': seg_res,
                    'branches': valid_branches}

    def update_graph(self, path):
        # Inverse sampling from root to leaves
        for i in range(len(path) - 1, 0, -1):
            self.bi_add(tuple(self.root + path[i-1]), tuple(self.root + path[i]))

    def bi_add(self, src, dst):
        self.graph[src].append(dst)
        self.graph[dst].append(src)

    def push(self, pts, positive=True):
        if isinstance(pts, list):
            pts = np.array(pts)
        if len(pts.shape) == 1:
            pts = pts[None, :]

        if positive: 
            self.pos = np.concatenate([self.pos, pts], axis=0)
        else:
            self.neg = np.concatenate([self.neg, pts], axis=0)
    
    def add_queue(self, pt):
        
        if self.pos.shape[0] == 0:
            self.pos = np.concatenate([self.pos, np.array([pt])], axis=0)
            self.queue.append(pt)
            return 
        dist = np.linalg.norm(self.pos - pt, axis=1)
        self.pos = np.concatenate([self.pos[dist >= self.patch_size.max() / self.sampling_grid], np.array([pt])], axis=0)
        
        if len(self.queue) == 0:
            self.queue.append(pt)
            return
        q_dist = np.linalg.norm(np.array(self.queue) - pt, axis=1)
        erase = np.where(q_dist < self.patch_size.max() / self.sampling_grid)
        print(f"Erasing queries {erase} ")
        for i, index in enumerate(erase):
            self.queue.pop(index - i)
        self.queue.append(pt) 

    def generate_boxes(self, prompt, steps=200, null_threshold=0.95, score_threshold=0.4, batch_size = 4):
        # Generate boxes, across all dimansio. 
        sampling_shape = self.image.shape - self.patch_size - 1
        x = np.linspace(0, sampling_shape[0], steps).astype(int)
        y = np.linspace(0, sampling_shape[1], steps).astype(int)
        mesh = np.stack(np.meshgrid(x, y), axis=2).reshape(-1, 2)
        
        