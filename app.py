import cv2
from matplotlib import pyplot as plt
import scipy
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw
import numpy as np
import time
import os
import sys

from config import AppConfig
import utils
import visualization as viz
import backend

CFG = AppConfig()
os.environ["CUDA_VISIBLE_DEVICES"] = CFG.CUDA_VISIBLE_DEVICES
sys.path.append(os.path.abspath("."))

def main():
    st.set_page_config(layout="wide", page_title="River Auto-Crawler Modular")
    st.title("Z-River: Zero-Shot Stream Segmentation for Riverbank Erosion Detection")

    # 1. Khởi tạo Model 
    if "model" not in st.session_state:
        model = backend.load_sam_model(CFG)
        model.read(CFG.IMAGE_PATH)
        st.session_state.model = model

    model = st.session_state.model

    # 2. Quản lý State 
    if "global_mask" not in st.session_state:
        full_img_init = Image.open(CFG.IMAGE_PATH).convert("RGB")
        real_w, real_h = full_img_init.size
        st.session_state.update({
            "global_mask": np.zeros((real_h, real_w), dtype=np.uint8),
            "pos_history": [],
            "neg_history": [],
            "is_running": False,
            "last_visual": None,
            "iter_count": 0  # Lưu biến đếm ở đây
        })

    # Sidebar điều khiển
    with st.sidebar:
        st.header("🎮 Điều khiển")
        mode = st.radio("Chế độ Click", ["Thêm điểm Positive (Sông)", "Thêm điểm Negative (Đất)"])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Chạy", use_container_width=True):
                st.session_state.is_running = True
        with col2:
            if st.button("⏸️ Dừng", use_container_width=True):
                st.session_state.is_running = False

        if st.button("🔄 Reset Tất cả", use_container_width=True):
            model.reset()
            model.read(CFG.IMAGE_PATH)
            st.session_state.global_mask.fill(0)
            st.session_state.pos_history = []
            st.session_state.neg_history = []
            st.session_state.is_running = False
            st.session_state.last_visual = None
            st.session_state.iter_count = 0 # Reset biến đếm
            st.rerun()

    # Layout chính
    col_input, col_view = st.columns(2)

    with col_input:
        st.subheader("Bản đồ tương tác")
        full_img = Image.open(CFG.IMAGE_PATH).convert("RGB")
        img_display, scale = viz.create_map_overlay(full_img, st.session_state.global_mask, CFG.DISPLAY_WIDTH)

        draw = ImageDraw.Draw(img_display)
        for gx, gy in st.session_state.pos_history:
            sx, sy = int(gx * scale), int(gy * scale)
            draw.ellipse((sx-4, sy-4, sx+4, sy+4), fill="#00FF00", outline="white")
        for gx, gy in st.session_state.neg_history:
            sx, sy = int(gx * scale), int(gy * scale)
            draw.ellipse((sx-4, sy-4, sx+4, sy+4), fill="#FF0000", outline="white")

        coords = streamlit_image_coordinates(img_display, key="input_map")
        
        if coords:
            gx, gy = int(coords['x'] / scale), int(coords['y'] / scale)
            if mode == "Thêm điểm Positive (Sông)":
                if (gx, gy) not in st.session_state.pos_history:
                    st.session_state.pos_history.append((gx, gy))
                    model.add_queue({'pt': np.array([gy, gx])}, prior=-1e9, isroot=True) 
                    st.toast(f"Đã nhận điểm ({gx}, {gy}) - Đang xử lý...", icon="⏳")
            else:
                if (gx, gy) not in st.session_state.neg_history:
                    st.session_state.neg_history.append((gx, gy))
                    model.neg = np.concatenate([model.neg, np.array([[gy, gx]])], axis=0) 
                    st.toast(f"Đã nhận điểm ({gx}, {gy}) - Đang xử lý...", icon="⏳")

    with col_view:
        st.subheader("Cửa sổ Segment chi tiết")
        view_ph = st.empty() 

        if st.session_state.last_visual is not None:
            view_ph.image(st.session_state.last_visual, use_container_width=True)

        # Vòng lặp xử lý không nhấp nháy
        if st.session_state.is_running and len(model.queue) > 0:
            # Chạy 1 lần cập nhật visual luôn
            for _ in range(1):
                if not st.session_state.is_running or len(model.queue) == 0:
                    break
                    
                output = model.iter(debug=True) 
                
                if output.get('ret'):
                    img_float = output['infer']['input'].copy().astype(float) / 255.0
                    input_mask = scipy.ndimage.morphological_gradient(output['infer']['inp_mask'], size=3)
                    src_y, src_x = model.root
                    
                    label_patch = st.session_state.global_mask[src_y:src_y+CFG.PATCH_SIZE, src_x:src_x+CFG.PATCH_SIZE]
                    label_mask = scipy.ndimage.morphological_gradient(label_patch, size=3)
                    pred_mask = scipy.ndimage.morphological_gradient(output['beta'].copy(), size=3)

                    combined_masks = np.stack([input_mask, label_mask, pred_mask], axis=-1).max(axis=-1)[..., None]
                    vis_image = img_float * (1 - combined_masks) \
                                + input_mask[..., None] * np.array([1, 0, 0]) \
                                + label_mask[..., None] * np.array([0, 1, 0]) \
                                + pred_mask[..., None] * np.array([0, 0, 1])
                    vis_image = (vis_image * 255).astype(np.uint8)

                    cmap = plt.get_cmap('hsv')
                    branches = [np.array(branch) for (cost, branch) in output.get('branches', [])]
                    for i_b, branch in enumerate(branches):
                        c_val = cmap(0.1 + 0.9 * (float(i_b) / max(1, len(branches))))
                        color_bgr = (int(c_val[2]*255), int(c_val[1]*255), int(c_val[0]*255))
                        cv2.polylines(vis_image, [branch[:, ::-1]], False, color_bgr, 1)

                    view_ph.image(vis_image, caption=f"Bước tự động: {st.session_state.iter_count}", use_container_width=True)
                    st.session_state.last_visual = vis_image
                    
                    viz.stitch_global_mask( 
                        st.session_state.global_mask, 
                        output.get('beta'), 
                        (src_x, src_y, src_x + CFG.PATCH_SIZE, src_y + CFG.PATCH_SIZE),
                        (CFG.PATCH_SIZE, CFG.PATCH_SIZE)
                    )
                    
                    st.session_state.iter_count += 1
                    time.sleep(0.01)
            
            # check queue mới
            st.rerun()

if __name__ == "__main__":
    main()


