import os, io, base64, json, gc, threading, time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
# SUSTech-SYSU dataset "category" column:
#   0 = Point-like   (small, scattered punctate staining — mildest)
#   1 = Point-Flaky  (mixed punctate + patchy staining — intermediate)
#   2 = Flaky        (large confluent patch(es) — most severe pattern)
CLASS_NAMES = ['Point-like', 'Point-Flaky', 'Flaky']
NUM_CLASSES = len(CLASS_NAMES)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH  = os.environ.get('MODEL_PATH', 'keratitis_model_final.pth')

# How many seconds of inactivity before the model is evicted from memory.
# Override via env var: MODEL_IDLE_TIMEOUT=300
IDLE_TIMEOUT = int(os.environ.get('MODEL_IDLE_TIMEOUT', 300))   # default 5 min

# ── Model manager (lazy load + idle eviction) ─────────────────────────────────
_model_lock      = threading.Lock()
_classifier      = None          # live model, or None when evicted
_last_used_at    = 0.0           # epoch-seconds of last request
_eviction_timer  = None          # background Timer handle


def _free_model():
    """Drop the model and reclaim memory. Called from the eviction timer."""
    global _classifier, _last_used_at, _eviction_timer
    with _model_lock:
        if _classifier is None:
            return
        # Only evict if we have truly been idle long enough
        if time.time() - _last_used_at < IDLE_TIMEOUT:
            _reschedule_eviction()
            return
        _classifier = None
        _eviction_timer = None
    # Release memory outside the lock so we don't hold it during GC
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[INFO] Model evicted after {IDLE_TIMEOUT}s of inactivity — memory freed.")


def _reschedule_eviction():
    """(Re)start the idle eviction timer. Must be called while holding _model_lock."""
    global _eviction_timer
    if _eviction_timer is not None:
        _eviction_timer.cancel()
    _eviction_timer = threading.Timer(IDLE_TIMEOUT, _free_model)
    _eviction_timer.daemon = True
    _eviction_timer.start()


def _build_mobilenet(num_classes, hidden=256, dropout=0.2):
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(m.last_channel, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden, num_classes),
    )
    return m


def _load_model():
    """Load checkpoint from disk and return a ready model. No side-effects."""
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    hidden_size      = state_dict['classifier.1.weight'].shape[0]
    num_classes_ckpt = state_dict['classifier.4.weight'].shape[0]
    assert num_classes_ckpt == NUM_CLASSES, (
        f"Checkpoint has {num_classes_ckpt} classes but CLASS_NAMES has {NUM_CLASSES}"
    )

    model = _build_mobilenet(num_classes_ckpt, hidden=hidden_size)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print(f"[INFO] Model loaded — classes: {CLASS_NAMES}, device: {DEVICE}")
    return model


def get_classifier():
    """
    Return the live model, loading it from disk if it was evicted.
    Resets the idle eviction timer on every call.
    Thread-safe.
    """
    global _classifier, _last_used_at
    with _model_lock:
        _last_used_at = time.time()
        if _classifier is None:
            print("[INFO] Model not in memory — loading from disk …")
            _classifier = _load_model()
        _reschedule_eviction()
        return _classifier


# ── GradCAM++ (pure PyTorch, no opencv) ───────────────────────────────────────
def gradcam_plusplus(model, tensor, target_idx):
    """Returns a (H, W) numpy heatmap in [0,1]."""
    features, grads = {}, {}

    target_layer = model.features[-1][0]

    def fwd_hook(m, inp, out):
        features['act'] = out

    def bwd_hook(m, gin, gout):
        grads['act'] = gout[0]

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    output = model(tensor)
    model.zero_grad()
    output[0, target_idx].backward()

    fh.remove()
    bh.remove()

    A  = features['act'][0]          # (C, H, W)
    dY = grads['act'][0]             # (C, H, W)

    # GradCAM++ alpha weights
    dY2   = dY ** 2
    dY3   = dY ** 3
    sum_A = A.sum(dim=(1, 2), keepdim=True)
    alpha = dY2 / (2 * dY2 + sum_A * dY3 + 1e-7)
    alpha = torch.where(dY > 0, alpha, torch.zeros_like(alpha))

    weights = (alpha * F.relu(dY)).sum(dim=(1, 2))   # (C,)
    cam     = (weights[:, None, None] * A).sum(dim=0) # (H, W)
    cam     = F.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-7)

    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(IMG_SIZE, IMG_SIZE),
        mode='bilinear', align_corners=False,
    )[0, 0]

    return cam_up.detach().cpu().numpy()


def apply_heatmap(rgb_np, cam):
    """Overlay jet colormap heatmap on rgb image (both float32 [0,1])."""
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    heatmap = np.stack([r, g, b], axis=-1)
    overlay = 0.5 * rgb_np + 0.5 * heatmap
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


# ── Transforms ────────────────────────────────────────────────────────────────
val_tfm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(pil_img, threshold=0.45):
    classifier = get_classifier()          # lazy-load + reset idle timer

    pil_img = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    raw_np  = np.array(pil_img) / 255.0
    tensor  = val_tfm(pil_img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    with torch.no_grad():
        logits = classifier(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx    = int(probs.argmax())
    confidence  = float(probs.max())
    class_probs = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}

    # GradCAM++ (needs grad)
    tensor2       = val_tfm(pil_img).unsqueeze(0).to(DEVICE)
    grayscale_cam = gradcam_plusplus(classifier, tensor2, pred_idx)

    overlay = apply_heatmap(raw_np.astype(np.float32), grayscale_cam)

    # Pseudo-segmentation
    mask   = (grayscale_cam > threshold).astype(np.uint8)
    coords = np.where(mask == 1)
    if len(coords[0]) > 0:
        y1, y2 = int(coords[0].min()), int(coords[0].max())
        x1, x2 = int(coords[1].min()), int(coords[1].max())
        pad  = 10
        bbox = (max(0, x1 - pad), max(0, y1 - pad),
                min(IMG_SIZE - 1, x2 + pad), min(IMG_SIZE - 1, y2 + pad))
    else:
        bbox = (0, 0, IMG_SIZE - 1, IMG_SIZE - 1)

    ulcer_area_pct = float(mask.sum()) / (IMG_SIZE * IMG_SIZE) * 100

    if ulcer_area_pct < 5:
        severity_label, severity_score = 'Mild', 1
    elif ulcer_area_pct < 15:
        severity_label, severity_score = 'Moderate', 2
    else:
        severity_label, severity_score = 'Severe', 3

    overlay_pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_pil.save(buf, format='PNG')
    overlay_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        'predicted_class': CLASS_NAMES[pred_idx],
        'confidence'     : round(confidence, 4),
        'class_probs'    : class_probs,
        'heatmap_b64'    : overlay_b64,
        'ulcer_bbox'     : bbox,
        'ulcer_area_pct' : round(ulcer_area_pct, 2),
        'severity_label' : severity_label,
        'severity_score' : severity_score,
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    try:
        pil_img = Image.open(file.stream)
        result  = predict(pil_img)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    with _model_lock:
        model_in_memory = _classifier is not None
        idle_for = round(time.time() - _last_used_at, 1) if _last_used_at else None
    return jsonify({
        'status'         : 'ok',
        'device'         : str(DEVICE),
        'classes'        : CLASS_NAMES,
        'model_in_memory': model_in_memory,
        'idle_seconds'   : idle_for,
        'idle_timeout'   : IDLE_TIMEOUT,
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)