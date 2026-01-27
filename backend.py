# model.py
import sys
import os
import streamlit as st
import numpy as np
from config import AppConfig
from utils import ensure_file

sys.path.append(os.path.abspath("."))
from src.zero_shot.sam import SamInferer

@st.cache_resource
def load_sam_model(cfg: AppConfig):
    ensure_file(cfg.IMAGE_PATH, cfg.URL_IMAGE)
    ensure_file(cfg.CHECKPOINT_PATH, cfg.URL_CHECKPOINT)
    ensure_file(cfg.CONFIG_PATH, cfg.URL_CONFIG)
    
    kwargs = cfg.MODEL_PARAMS.copy()
    kwargs.update({
        'cfg': os.path.abspath(cfg.CONFIG_PATH),
        'ckpt': os.path.abspath(cfg.CHECKPOINT_PATH),
        'patch_size': [cfg.PATCH_SIZE, cfg.PATCH_SIZE]
    })
    
    return SamInferer(**kwargs)

def calculate_next_jump_vector(output: dict, patch_size: int):
    """Tính toán vector nhảy tiếp theo dựa trên Output của model."""
    roots = output.get('roots', {})
    if 'pts' not in roots or len(roots['pts']) == 0:
        return None, False

    best_vec = None
    max_score = -1
    found = False
    margin = 20

    for i in range(len(roots['pts'])):
        pt = roots['pts'][i]
        vec = roots['directions'][i]
        score = np.linalg.norm(vec)

        # Dự đoán điểm đến
        projected_pt = pt + vec * 50
        
        # Kiểm tra xem có văng ra ngoài biên (Out of bounds) không?
        is_out = (
            projected_pt[0] < margin or projected_pt[0] > patch_size - margin or
            projected_pt[1] < margin or projected_pt[1] > patch_size - margin
        )

        # Logic chọn hướng: Hướng mạnh + Đi ra ngoài biên
        if score > 0.5 and is_out and score > max_score:
            max_score = score
            best_vec = vec * 350 # Scale lực nhảy
            found = True
            
    return best_vec, found