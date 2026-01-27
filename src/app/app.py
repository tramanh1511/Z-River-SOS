"""
CVAT-compatible SAM2 segmentation microservice
--------------------------------------------
This Flask app exposes simple endpoints that a CVAT frontend (or any web client)
can call to run Segment-Anything v2 (SAM2) predictions with:
 - positive / negative clicks
 - zooming (by providing a crop or scale)
 - undo / redo (per-session stacks)

How to use:
 1. Replace the `load_sam2_model()` placeholder with your real SAM2 model init.
 2. Run: python cvat_sam2_service.py
 3. POST /segment with multipart/form-data or JSON payload.

Endpoints:
 - POST /create_session -> {session_id}
 - POST /segment -> {mask (base64 PNG), bbox, score}
 - POST /undo -> last result popped
 - POST /redo -> redo last undone
 - POST /reset -> clear stacks

This file intentionally keeps dependencies minimal and focuses on a
clear API that CVAT can call from a custom plugin or external tool.

Note: This is a demo scaffold. Test and adapt for production (auth, rate
limits, GPU assignment, model concurrency, persistent storage).
"""

from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw
import io
import base64
import numpy as np
import uuid
import threading
import traceback
import os
import hydra

import rootutils
rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
from src.zero_shot.sam import SamInferer
app = Flask(__name__)
# Simple in-memory session store. For production, use Redis or DB.
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()

# Each session: {
#   'image': PIL.Image,           # original image
#   'stacks': [result1, ...],     # history stack (list of dicts)
#   'undone': [resultA, ...]      # redo stack
# }
# Each result: { 'mask': numpy array HxW bool, 'visual': bytes PNG, 'meta': {...} }


# --------------------------- Placeholder SAM2 loader ---------------------------
# Replace this with the real SAM2 model loader / inference API you have.
# Example pseudo-code for model usage:
#   model = load_sam2_model('sam_v2_checkpoint.pth')
#   mask = model.predict(image, positive_points, negative_points, crop=None)
# For this demo we'll implement a dummy "segmentation" that draws circles around
# positive clicks and subtracts negative clicks. This allows the demo UI to test
# undo/redo/zoom behavior without a real model.

with hydra.compose("configs/model/sam.yaml", return_hydra_config=True) as cfg:
    MODEL = hydra.utils.instantiate(cfg)

# --------------------------- Utilities ---------------------------

def pil_to_base64_png(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def mask_to_visual_overlay(image_pil, mask_np, alpha=0.5):
    """Return PNG bytes of the image with mask overlay for quick preview."""
    overlay = image_pil.convert('RGBA')
    mask_img = Image.fromarray((mask_np * 255).astype('uint8'))
    mask_col = Image.new('RGBA', image_pil.size, (255, 0, 0, int(255 * alpha)))
    # composite using mask
    overlay.paste(mask_col, (0, 0), mask_img)
    buf = io.BytesIO()
    overlay.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


def ensure_session(session_id):
    with SESSIONS_LOCK:
        if session_id not in SESSIONS:
            raise KeyError('Unknown session: {}'.format(session_id))
        return SESSIONS[session_id]


# --------------------------- API Endpoints ---------------------------

@app.route('/create_session', methods=['POST'])
def create_session():
    """Create a new session and upload the source image.
    Accepts multipart/form-data with 'image' file, or JSON with 'image_b64'.
    Returns: {session_id}
    """
    try:
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file.stream).convert('RGB')
        else:
            data = request.json or {}
            if 'image_b64' in data:
                image = Image.open(io.BytesIO(base64.b64decode(data['image_b64']))).convert('RGB')
            else:
                return jsonify({'error': 'Please upload image file or image_b64 in JSON'}), 400

        session_id = str(uuid.uuid4())
        with SESSIONS_LOCK:
            SESSIONS[session_id] = {
                'image': image,
                'stacks': [],
                'undone': []
            }
        return jsonify({'session_id': session_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/segment', methods=['POST'])
def segment():
    """Run segmentation with clicks and optional zoom/crop.
    Body (JSON): {
      session_id: str,
      positive: [[x,y], ...],
      negative: [[x,y], ...],
      crop: [x1,y1,x2,y2]  # optional crop in original image coords
    }
    Response: {mask_b64_png, overlay_b64_png, score, bbox}
    """
    try:
        data = request.get_json(force=True)
        session_id = data['session_id']
        pos = data.get('positive', [])
        neg = data.get('negative', [])
        crop = data.get('crop', None)

        sess = ensure_session(session_id)
        image = sess['image']

        # If crop provided, crop image and offset points
        offset_x = offset_y = 0
        if crop:
            x1, y1, x2, y2 = map(int, crop)
            # clamp
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(image.width, x2); y2 = min(image.height, y2)
            crop_box = (x1, y1, x2, y2)
            image_crop = image.crop(crop_box)
            offset_x, offset_y = x1, y1
            # shift points
            pos_shift = [(int(x - offset_x), int(y - offset_y)) for x, y in pos]
            neg_shift = [(int(x - offset_x), int(y - offset_y)) for x, y in neg]
            mask_np, score = MODEL.predict(image_crop, pos_shift, neg_shift, crop_box=crop_box)
            # expand mask back to full image size
            full_mask = np.zeros((image.height, image.width), dtype=bool)
            mh, mw = mask_np.shape
            full_mask[y1:y1+mh, x1:x1+mw] = mask_np
            mask_np = full_mask
        else:
            mask_np, score = MODEL.predict(image, pos, neg, crop_box=None)

        # compute bbox of mask
        ys, xs = np.where(mask_np)
        if len(xs) == 0:
            bbox = None
        else:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        visual_bytes = mask_to_visual_overlay(image, mask_np)
        result = {
            'mask': mask_np,
            'visual': visual_bytes,
            'meta': {
                'score': float(score),
                'bbox': bbox,
                'positive': pos,
                'negative': neg,
                'crop': crop
            }
        }

        # push to history and clear redo stack
        with SESSIONS_LOCK:
            sess['stacks'].append(result)
            sess['undone'].clear()

        return jsonify({
            'mask_b64_png': base64.b64encode(Image.fromarray((mask_np * 255).astype('uint8')).tobytes()).decode('ascii'),
            # Note: the above is raw pixels; clients usually prefer overlay or full PNG
            'overlay_b64_png': base64.b64encode(visual_bytes).decode('ascii'),
            'score': float(score),
            'bbox': bbox
        })

    except KeyError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/undo', methods=['POST'])
def undo():
    """Pop last segmentation result and push to undone stack. Returns the new top result or empty."""
    try:
        data = request.get_json(force=True)
        session_id = data['session_id']
        sess = ensure_session(session_id)
        with SESSIONS_LOCK:
            if not sess['stacks']:
                return jsonify({'ok': False, 'message': 'nothing to undo'})
            item = sess['stacks'].pop()
            sess['undone'].append(item)
            top = sess['stacks'][-1] if sess['stacks'] else None

        if top is None:
            return jsonify({'ok': True, 'top': None})

        return jsonify({
            'ok': True,
            'top_overlay_b64': base64.b64encode(top['visual']).decode('ascii'),
            'meta': top['meta']
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/redo', methods=['POST'])
def redo():
    """Redo last undone result."""
    try:
        data = request.get_json(force=True)
        session_id = data['session_id']
        sess = ensure_session(session_id)
        with SESSIONS_LOCK:
            if not sess['undone']:
                return jsonify({'ok': False, 'message': 'nothing to redo'})
            item = sess['undone'].pop()
            sess['stacks'].append(item)
            top = item

        return jsonify({
            'ok': True,
            'top_overlay_b64': base64.b64encode(top['visual']).decode('ascii'),
            'meta': top['meta']
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    try:
        data = request.get_json(force=True)
        session_id = data['session_id']
        sess = ensure_session(session_id)
        with SESSIONS_LOCK:
            sess['stacks'].clear()
            sess['undone'].clear()
        return jsonify({'ok': True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/get_overlay/<session_id>', methods=['GET'])
def get_overlay(session_id):
    try:
        sess = ensure_session(session_id)
        top = sess['stacks'][-1] if sess['stacks'] else None
        if top is None:
            return jsonify({'ok': False, 'message': 'no overlay'})
        return send_file(io.BytesIO(top['visual']), mimetype='image/png')
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --------------------------- Simple CVAT frontend snippet ---------------------------
# The following is a tiny example of how the CVAT UI (or any web client)
# could call this service. Keep it in comments for implementers.
#
# fetch('/create_session', {method:'POST', body: formData}) -> {session_id}
# fetch('/segment', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({session_id, positive:[[x,y]], negative:[[x,y]], crop:[x1,y1,x2,y2]})})
#   -> {overlay_b64_png, score, bbox}
# fetch('/undo', {method:'POST', body: JSON.stringify({session_id})})
# fetch('/redo', {method:'POST', body: JSON.stringify({session_id})})

# --------------------------- Run server ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print('Starting CVAT SAM2 demo service on port', port)
    app.run(host='0.0.0.0', port=port, debug=True)
