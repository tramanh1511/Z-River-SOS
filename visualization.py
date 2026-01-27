# visualization.py
import numpy as np
import cv2
from PIL import Image
from utils import process_mask_to_u8

def render_visualization_frame(patch_img_pil: Image.Image, output: dict, patch_size: int) -> np.ndarray:
    """Vẽ frame hiển thị: Ảnh gốc + Viền đỏ/xanh + Skeleton + Mũi tên."""
    target_hw = (patch_size, patch_size)
    image = np.array(patch_img_pil.resize(target_hw)).copy()
    
    kernel = np.ones((3,3), np.uint8)
    def get_edge(m):
        return cv2.subtract(cv2.dilate(m, kernel, 1), cv2.erode(m, kernel, 1))

    # 1. Vẽ Viền Mask (Input & Prediction)
    inp_mask = process_mask_to_u8(output['infer'].get('inp_mask'), target_hw)
    pred_mask = process_mask_to_u8(output.get('beta'), target_hw)
    
    image[get_edge(inp_mask) > 0] = [255, 0, 0]   # Input: Đỏ
    image[get_edge(pred_mask) > 0] = [0, 0, 255]  # Pred: Xanh dương

    # 2. Vẽ Skeleton (Xương)
    if 'skeleton' in output:
        skel = process_mask_to_u8(output['skeleton'], target_hw)
        # Fix lỗi skeleton bị đảo ngược màu
        if np.mean(skel > 0) > 0.4: skel = cv2.bitwise_not(skel)
        if np.mean(skel > 0) < 0.4:
            skel_dilated = cv2.dilate(skel, kernel, iterations=1)
            image[skel_dilated > 0] = [255, 255, 0] # Vàng

    # 3. Vẽ Mũi tên hướng dòng chảy (Roots)
    roots = output.get('roots', {})
    if 'pts' in roots and len(roots['pts']) > 0:
        pts, dirs = roots['pts'], roots['directions']
        for i in range(len(pts)):
            px, py = int(pts[i][0]), int(pts[i][1])
            dx, dy = dirs[i]
            
            # Chỉ vẽ nếu nằm trong ảnh
            if 0 <= px < patch_size and 0 <= py < patch_size:
                cv2.circle(image, (px, py), 5, (0, 255, 0), 2)
                if np.linalg.norm([dx, dy]) > 0:
                    end_pt = (int(px + dx * 40), int(py + dy * 40))
                    cv2.arrowedLine(image, (px, py), end_pt, (0, 255, 0), 2, tipLength=0.3)

    return image.astype(np.uint8)

def stitch_global_mask(global_mask, local_mask_raw, roi_coords, patch_size_wh):
    """Ghép mask con vào mask tổng."""
    nx1, ny1, nx2, ny2 = roi_coords
    local_u8 = process_mask_to_u8(local_mask_raw, patch_size_wh)
    valid_local = local_u8[0:ny2-ny1, 0:nx2-nx1]
    
    current_global = global_mask[ny1:ny2, nx1:nx2]
    if current_global.shape == valid_local.shape:
        global_mask[ny1:ny2, nx1:nx2] = cv2.bitwise_or(current_global, valid_local)

def create_map_overlay(full_img_pil, global_mask, display_width):
    """Tạo ảnh bản đồ có lớp phủ màu xanh."""
    w, h = full_img_pil.size
    scale = display_width / w
    new_h = int(h * scale)
    
    img_display = full_img_pil.resize((display_width, new_h))
    mask_small = cv2.resize(global_mask, img_display.size, interpolation=cv2.INTER_NEAREST)
    
    if np.sum(mask_small) > 0:
        overlay = np.array(img_display)
        overlay[mask_small > 0] = [0, 255, 0] 
        blended = cv2.addWeighted(np.array(img_display), 0.7, overlay, 0.3, 0)
        return Image.fromarray(blended), scale
    
    return img_display, scale