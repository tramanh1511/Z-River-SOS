# app.py
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

    model = backend.load_sam_model(CFG)
    
    if not os.path.exists(CFG.IMAGE_PATH):
        st.warning("Đang tải ảnh demo...")
        utils.ensure_file(CFG.IMAGE_PATH, CFG.URL_IMAGE)
        
    full_img = Image.open(CFG.IMAGE_PATH).convert("RGB")
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
            
            # Jump loop 
            # Code gốc Hưng để auto jump từ batch đã biết, còn đây chọn thủ công điểm để segment @@ ._. 
            for jump in range(CFG.MAX_WINDOW_JUMPS):
                status_ph.info(f"🔄 Jump {jump+1}/{CFG.MAX_WINDOW_JUMPS}: Xử lý vùng ({curr_cx}, {curr_cy})...")
                
                # Cắt Patch ảnh
                nx1 = max(0, curr_cx - CFG.PATCH_SIZE // 2); ny1 = max(0, curr_cy - CFG.PATCH_SIZE // 2)
                nx2 = min(real_w, nx1 + CFG.PATCH_SIZE); ny2 = min(real_h, ny1 + CFG.PATCH_SIZE)
                
                crop_pil = full_img.crop((nx1, ny1, nx2, ny2))
                # Padding nếu ở biên
                if crop_pil.size != (CFG.PATCH_SIZE, CFG.PATCH_SIZE):
                    temp = Image.new("RGB", (CFG.PATCH_SIZE, CFG.PATCH_SIZE), (0,0,0))
                    temp.paste(crop_pil, (0,0))
                    crop_pil = temp
                crop_pil.save(CFG.TEMP_CROP_PATH)
                
                model.read(CFG.TEMP_CROP_PATH)
                model.label = np.zeros((CFG.PATCH_SIZE, CFG.PATCH_SIZE), dtype=np.uint8)
                center_pt = np.array([CFG.PATCH_SIZE//2, CFG.PATCH_SIZE//2])
                model.add_queue(utils.PromptWrapper({'pt': utils.SafePoint(center_pt)}), isroot=True)
                
                next_move_vec = None
                
                for step in range(CFG.LOCAL_STEPS):
                    if len(model.queue) == 0: break
                    out = model.iter(debug=True)
                    
                    if out.get('ret'):
                        vis_img = viz.render_visualization_frame(crop_pil, out, CFG.PATCH_SIZE)
                        view_ph.image(vis_img, caption=f"Jump:{jump+1} Step:{step}", use_container_width=True)
                        st.session_state["last_visual"] = vis_img
                        
                        # Ghép mask
                        viz.stitch_global_mask(
                            st.session_state["global_mask"], 
                            out.get('beta'), 
                            (nx1, ny1, nx2, ny2), 
                            (CFG.PATCH_SIZE, CFG.PATCH_SIZE)
                        )
                        
                        # model: Tính toán nhảy
                        vec, found = backend.calculate_next_jump_vector(out, CFG.PATCH_SIZE)
                        if found: next_move_vec = vec
                        
                        time.sleep(0.01) 

                if next_move_vec is not None:
                    curr_cx = int(max(0, min(real_w, curr_cx + next_move_vec[0])))
                    curr_cy = int(max(0, min(real_h, curr_cy + next_move_vec[1])))
                else:
                    status_ph.warning(f"Dừng tại Jump {jump+1} (Hết dòng chảy).")
                    break
            
            status_ph.success("Done")
            time.sleep(0.5)
            st.rerun()

if __name__ == "__main__":
    main()