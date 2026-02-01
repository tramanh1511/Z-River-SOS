import cv2
from matplotlib import pyplot as plt
import scipy
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw, ImageOps
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

st.set_page_config(page_title="Z-River", layout="wide")

st.markdown(f"""
<style>
   header[data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0); 
        z-index: 100;
    }}

    .zr-header {{
        display: flex;
        align-items: center;
        background-color: #213448;
        padding: 5px 20px;
        
        position: fixed;
        top: 0;
        left: 0; 
        width: 100%;
        height: 60px;
        z-index: 99;
    }}

    [data-testid="stAppViewBlockContainer"] {{
        padding-top: 80px !important; 
    }}

    .zr-header img {{
        height: 35px;
        margin-right: 15px;
    }}
    .zr-text {{ color: white; }}
    .zr-title {{ font-size: 20px; font-weight: 800; line-height: 1; }}
    .zr-subtitle {{ font-size: 11px; opacity: 0.8; }}
    .zr-icon {{
        font-size: 30px;
        margin-right: 15px;
        display: flex;
        align-items: center;
    }}
</style>

<div class="zr-header">
    <span class="zr-icon">🌊</span>
    <div class="zr-text">
        <div class="zr-title">Z-River</div>
        <div class="zr-subtitle">Zero-Shot Stream Segmentation for Riverbank Erosion Detection</div>
    </div>
</div>
""", unsafe_allow_html=True)

def upload_page():
    st.title("Upload River Image")
    st.markdown("""
    <style>
    .stButton {
        display: flex;
        justify-content: center;
    }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff3333;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded:
            img = Image.open(uploaded)
            img = ImageOps.exif_transpose(img) 
            img.save("./imgs/image.png", optimize=False, compress_level=0)
            img = img.convert("RGB")
            m_col1, m_col2, m_col3 = st.columns([1, 2, 1])
            with m_col2:
                if st.button("Start Segmentation...", use_container_width=True):
                    st.session_state["full_img"] = img
                    st.session_state["full_img_path"] = "./imgs/image.png"
                    st.session_state["page"] = "main"
                    st.rerun()
    with col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if uploaded:
            st.image(img, use_container_width=True)
        else:
            st.info("Please upload an image to see the preview.")


def main_page():
    st.title("Z-River: Zero-Shot Stream Segmentation for Riverbank Erosion Detection")

    if "model" not in st.session_state:
        model = backend.load_sam_model(CFG)
        model.read(st.session_state["full_img_path"])
        st.session_state.model = model

    model = st.session_state.model
    
    full_img = st.session_state["full_img"]
    real_w, real_h = full_img.size

    # Quản lý State
    if "global_mask" not in st.session_state:
        full_img_init = Image.open(CFG.IMAGE_PATH).convert("RGB")
        real_w, real_h = full_img_init.size
        st.session_state.update({
            "global_mask": np.zeros((real_h, real_w), dtype=np.uint8),
            "pos_history": [],
            "neg_history": [],
            "is_running": False,
            "last_visual": None,
            "iter_count": 0  
        })

    
    # with st.sidebar:
    #     st.header("🎮 Điều khiển")
    #     mode = st.radio("Chế độ Click", ["Thêm điểm Positive (Sông)", "Thêm điểm Negative (Đất)"])
        
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         if st.button("▶️ Chạy", use_container_width=True):
    #             st.session_state.is_running = True
    #     with col2:
    #         if st.button("⏸️ Dừng", use_container_width=True):
    #             st.session_state.is_running = False

    #     if st.button("🔄 Reset Tất cả", use_container_width=True):
    #         model.reset()
    #         model.read(CFG.IMAGE_PATH)
    #         st.session_state.global_mask.fill(0)
    #         st.session_state.pos_history = []
    #         st.session_state.neg_history = []
    #         st.session_state.is_running = False
    #         st.session_state.last_visual = None
    #         st.session_state.iter_count = 0 # Reset biến đếm
    #         st.rerun()
    
    with st.container():
        # Thêm một chút CSS để làm nổi bật khu vực điều khiển
        st.markdown("""
            <style>
            .control-box {
                background-color: #1e2b3b;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                border: 1px solid #34495e;
            }
            </style>
        """, unsafe_allow_html=True)

        # Chia cột để dàn hàng ngang các nút bấm
        c1, c2, c3, c4 = st.columns([2.5, 1, 1, 1.5])

        with c1:
            mode = st.radio(
                "Chế độ Click:", 
                ["Thêm điểm Positive (Sông)", "Thêm điểm Negative (Đất)"], 
                horizontal=True, 
                label_visibility="collapsed"
            )
        
        with c2:
            if st.button("▶️ Chạy", use_container_width=True):
                st.session_state.is_running = True
                # st.rerun()
                
        with c3:
            if st.button("⏸️ Dừng", use_container_width=True):
                st.session_state.is_running = False
                
        with c4:
            if st.button("🔄 Reset", use_container_width=True):
                model.reset()
                st.session_state.global_mask.fill(0)
                st.session_state.pos_history = []
                st.session_state.neg_history = []
                st.session_state.is_running = False
                st.session_state.last_visual = None
                st.session_state.iter_count = 0

                # st.session_state["page"] = "upload" 
                st.rerun()

    st.divider()
    col_input, col_view = st.columns([3, 2])
    with col_input:
        st.subheader("Bản đồ tương tác")
        img_display, scale = viz.create_map_overlay(
            full_img, st.session_state["global_mask"], CFG.DISPLAY_WIDTH
        )

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
            
            st.rerun()

def app():
    if "page" not in st.session_state:
        st.session_state["page"] = "upload"

    if st.session_state["page"] == "upload":
        upload_page()
    else:
        main_page()

if __name__ == "__main__":
    app()