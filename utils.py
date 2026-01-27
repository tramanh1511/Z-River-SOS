# utils.py
import os
import urllib.request
import numpy as np
import cv2

class SafePoint(np.ndarray):
    """Wrapper để giúp numpy array so sánh được trong PriorityQueue."""
    def __new__(cls, input_array): return np.asarray(input_array).view(cls)
    def __lt__(self, other): return id(self) < id(other)
    def __gt__(self, other): return id(self) > id(other)
    def __eq__(self, other): return id(self) == id(other)

class PromptWrapper(dict):
    """Wrapper cho dictionary."""
    def __lt__(self, other): return False 

def ensure_file(path: str, url: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)

def process_mask_to_u8(mask_data, target_size):
    if mask_data is None: 
        return np.zeros(target_size, dtype=np.uint8)
    
    m = np.array(mask_data)
    # Resize nếu kích thước không khớp
    if m.shape[:2] != target_size:
        m = cv2.resize(m.astype(float), (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8) * 255