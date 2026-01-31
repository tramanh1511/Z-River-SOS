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

    model = backend.load_sam_model(CFG)
    
    full_img = st.session_state["full_img"]
    real_w, real_h = full_img.size

    # Quản lý State
    if "global_mask" not in st.session_state:
        st.session_state.update({
            "global_mask": np.zeros((real_h, real_w), dtype=np.uint8),
            "click_history": [],
            "last_visual": None,
            "processing_queue": None
        })

    col_input, col_view = st.columns(2)

    with col_input:
        st.subheader("Image")
        
        img_display, scale = viz.create_map_overlay(
            full_img, st.session_state["global_mask"], CFG.DISPLAY_WIDTH
        )

        draw = ImageDraw.Draw(img_display)
        for gx, gy in st.session_state["click_history"]:
            sx, sy = int(gx * scale), int(gy * scale)
            draw.ellipse((sx-4, sy-4, sx+4, sy+4), fill="red", outline="white")

        # Component tương tác
        coords = streamlit_image_coordinates(img_display, key="input_map")
        
        if st.button("Reset All", use_container_width=True):
            st.session_state["global_mask"] = np.zeros((real_h, real_w), dtype=np.uint8)
            st.session_state["click_history"] = []
            st.session_state["last_visual"] = None
            st.session_state["processing_queue"] = None
            st.session_state["page"] = "upload"
            model.reset()
            st.rerun()

        # Click
        if coords:
            gx, gy = int(coords['x'] / scale), int(coords['y'] / scale)
            last_pt = st.session_state["click_history"][-1] if st.session_state["click_history"] else (-1, -1)

            if (gx, gy) != last_pt:
                st.session_state["click_history"].append((gx, gy))
                st.session_state["processing_queue"] = (gx, gy)
                model.reset() 
                st.toast(f"Đã nhận điểm ({gx}, {gy}) - Đang xử lý...", icon="⏳")


    with col_view:
        st.subheader("Segmentation visualization")
        view_ph = st.empty()
        status_ph = st.empty()

        if st.session_state["last_visual"] is not None:
            view_ph.image(st.session_state["last_visual"], use_container_width=True)
        else:
            view_ph.info("Waiting for start point...")

        if st.session_state["processing_queue"]:
            curr_cx, curr_cy = st.session_state["processing_queue"]
            st.session_state["processing_queue"] = None 
            
            model.reset()
            model.read(st.session_state["full_img_path"])
            model.add_queue({'pt': np.array([curr_cy, curr_cx])}, isroot=True)

            for i in range(200):
                if len(model.queue) == 0:
                    break
                
                output = model.iter(debug=True)
                
                if output.get('ret'):
                    img_float = output['infer']['input'].copy().astype(float) / 255.0
                    
                    input_mask = scipy.ndimage.morphological_gradient(output['infer']['inp_mask'], size=3)
                    src_y, src_x = model.root
                    label_patch = st.session_state["global_mask"][src_y:src_y+CFG.PATCH_SIZE, src_x:src_x+CFG.PATCH_SIZE]
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
                        color_bgr = (int(c_val[2]*255), int(c_val[1]*255), int(c_val[0]*255)) # RGB to BGR
                        cv2.polylines(vis_image, [branch[:, ::-1]], False, color_bgr, 1)

                    annotation = output['infer'].get('pts', [])
                    a_label = output['infer'].get('label', [])
                    for pt, lab in zip(annotation, a_label):
                        pt_color = (0, 255, 0) if lab == 1 else (0, 0, 255) # Green for pos, Blue for neg
                        cv2.circle(vis_image, tuple(pt.astype(int)), 4, pt_color, -1)

                    roots_pts = output['roots'].get('pts', [])
                    roots_dirs = output['roots'].get('directions', [])
                    for pt, di in zip(roots_pts, roots_dirs):
                        cv2.circle(vis_image, pt[::-1].astype(int), 15, (0, 255, 255), 2) 
                        if np.linalg.norm(di) > 0:
                            root_dst = (pt + di * 60).astype(int)
                            cv2.arrowedLine(vis_image, pt[::-1].astype(int), root_dst[::-1].astype(int), (0, 255, 255), 2)

                    view_ph.image(vis_image, caption=f"Auto-Tracking Step {i + 1}", use_container_width=True)
                    st.session_state["last_visual"] = vis_image
                    
                    viz.stitch_global_mask(
                        st.session_state["global_mask"], 
                        output.get('beta'), 
                        (src_x, src_y, src_x + CFG.PATCH_SIZE, src_y + CFG.PATCH_SIZE),
                        (CFG.PATCH_SIZE, CFG.PATCH_SIZE)
                    )
                    time.sleep(0.01) 
            
            status_ph.success("Done")
            time.sleep(0.5)
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