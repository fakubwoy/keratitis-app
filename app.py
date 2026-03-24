import os, io, base64, json
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
CLASS_NAMES = ['Type-0', 'Type-1', 'Type-2']
NUM_CLASSES = len(CLASS_NAMES)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH  = os.environ.get('MODEL_PATH', 'keratitis_model_final.pth')

# ── Build model ───────────────────────────────────────────────────────────────
# The checkpoint was trained with a custom two-layer head:
#   classifier.0  Dropout
#   classifier.1  Linear(1280 → 256)
#   classifier.2  ReLU
#   classifier.3  Dropout
#   classifier.4  Linear(256 → num_classes)
# This must match exactly so load_state_dict() succeeds.

def build_mobilenet(num_classes, hidden=256, dropout=0.2):
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(m.last_channel, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden, num_classes),
    )
    return m

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

if 'model_state_dict' in ckpt:
    state_dict    = ckpt['model_state_dict']
    saved_classes = ckpt.get('class_names', CLASS_NAMES)
else:
    state_dict    = ckpt
    saved_classes = CLASS_NAMES

# Infer num_classes from the checkpoint's final layer
final_weight_key = [k for k in state_dict if k.endswith('.weight') and
                    any(f'classifier.{i}' in k for i in range(4, 10))]
if final_weight_key:
    num_classes_ckpt = state_dict[final_weight_key[-1]].shape[0]
else:
    num_classes_ckpt = len(saved_classes)

# Infer hidden size from classifier.1.weight
hidden_size = 256
if 'classifier.1.weight' in state_dict:
    hidden_size = state_dict['classifier.1.weight'].shape[0]

classifier = build_mobilenet(num_classes_ckpt, hidden=hidden_size)
classifier.load_state_dict(state_dict)

CLASS_NAMES = saved_classes
NUM_CLASSES = len(CLASS_NAMES)

classifier.to(DEVICE).eval()
print(f"[INFO] Model loaded — classes: {CLASS_NAMES}, device: {DEVICE}")

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

    A  = features['act'][0]   # (C, H, W)
    dY = grads['act'][0]      # (C, H, W)

    # GradCAM++ alpha weights
    dY2   = dY ** 2
    dY3   = dY ** 3
    sum_A = A.sum(dim=(1, 2), keepdim=True)
    alpha = dY2 / (2 * dY2 + sum_A * dY3 + 1e-7)
    alpha = torch.where(dY > 0, alpha, torch.zeros_like(alpha))

    weights = (alpha * F.relu(dY)).sum(dim=(1, 2))  # (C,)
    cam     = (weights[:, None, None] * A).sum(dim=0)  # (H, W)
    cam     = F.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-7)

    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(IMG_SIZE, IMG_SIZE),
        mode='bilinear', align_corners=False
    )[0, 0]

    return cam_up.detach().cpu().numpy()


def apply_heatmap(rgb_np, cam):
    """Overlay jet colormap heatmap on an RGB image (both float32 [0,1])."""
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
    pil_img = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    raw_np  = np.array(pil_img) / 255.0

    tensor = val_tfm(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = classifier(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx    = int(probs.argmax())
    confidence  = float(probs.max())
    class_probs = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}

    # GradCAM++ needs a fresh tensor with grad enabled
    tensor2       = val_tfm(pil_img).unsqueeze(0).to(DEVICE)
    grayscale_cam = gradcam_plusplus(classifier, tensor2, pred_idx)

    overlay = apply_heatmap(raw_np.astype(np.float32), grayscale_cam)

    # Pseudo-segmentation via CAM threshold
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
    return jsonify({'status': 'ok', 'device': str(DEVICE), 'classes': CLASS_NAMES})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)