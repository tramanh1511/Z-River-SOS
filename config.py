# config.py
import torch
from dataclasses import dataclass
import numpy as np
import os

@dataclass
class AppConfig:
    # --- System ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    CUDA_VISIBLE_DEVICES: str = "0" if torch.cuda.is_available() else ""
    
    # Đường dẫn input
    IMAGE_PATH: str = "image/image.png" 
    TEMP_CROP_PATH: str = "temp_auto_crawl.png"
    
    # Đường dẫn Model
    CHECKPOINT_PATH: str = "checkpoints/sam2.1_hiera_s.pt"
    CONFIG_PATH: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    # --- URLs ---
    URL_IMAGE: str = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
    URL_CHECKPOINT: str = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    URL_CONFIG: str = "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"

    # --- Algorithm Parameters ---
    PATCH_SIZE: int = 512
    MAX_WINDOW_JUMPS: int = 15     # Số bước nhảy tối đa
    LOCAL_STEPS: int = 15          # Số bước loang màu nội bộ
    
    # --- UI Settings ---
    DISPLAY_WIDTH: int = 800

    # --- SAM Model Params ---
    MODEL_PARAMS = {
        'roi': [128, 128],
        'root_area': 800,
        'max_roots': 3,
        'patience': 2,
        'beta': 0.65,
        'd_alpha': 0.1,
        'alpha': 0.25,
        'decay': 0.7,
        'fill_kernel_size': 5,
        'thresh': 0.95,
        'min_length': 50,
        'sampling_dis': 300,
        'back_off': 15,
        'neg_sampling_grid': 6,
        'neg_dis': 15,
        'pos_dis': 30,
        'pos_rad': 200,
        'pos_sc': 2.,
        'confidence': 0.9,
        'topk': 2,
        'stable_weight': 1.5,
        'post_act': False,
        'gamma': 2.,
        'label_bins': np.linspace(0, 1, num=50).tolist(),
    }