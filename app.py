import os, io, base64, json
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
CLASS_NAMES = ['Type-0', 'Type-1', 'Type-2']
NUM_CLASSES = len(CLASS_NAMES)
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH  = os.environ.get('MODEL_PATH', 'keratitis_model_final.pth')

# ── Load model ────────────────────────────────────────────────────────────────
def build_mobilenet(num_classes):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m

classifier = build_mobilenet(NUM_CLASSES)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
# handle both raw state_dict and wrapped checkpoint
if 'model_state_dict' in ckpt:
    classifier.load_state_dict(ckpt['model_state_dict'])
    saved_classes = ckpt.get('class_names', CLASS_NAMES)
    CLASS_NAMES   = saved_classes
    NUM_CLASSES   = len(CLASS_NAMES)
else:
    classifier.load_state_dict(ckpt)

classifier.to(DEVICE).eval()

# ── GradCAM++ ─────────────────────────────────────────────────────────────────
target_layer = [classifier.features[-1][0]]
cam = GradCAMPlusPlus(model=classifier, target_layers=target_layer)

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
    tensor  = val_tfm(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = classifier(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx   = int(probs.argmax())
    confidence = float(probs.max())
    class_probs = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}

    # GradCAM++
    cam_tensor = tensor.detach().requires_grad_(True)
    with torch.enable_grad():
        grayscale_cam = cam(input_tensor=cam_tensor,
                            targets=[ClassifierOutputTarget(pred_idx)])[0]
    overlay = show_cam_on_image(raw_np.astype(np.float32), grayscale_cam, use_rgb=True)

    # Pseudo-segmentation
    mask   = (grayscale_cam > threshold).astype(np.uint8)
    coords = np.where(mask == 1)
    if len(coords[0]) > 0:
        y1, y2 = int(coords[0].min()), int(coords[0].max())
        x1, x2 = int(coords[1].min()), int(coords[1].max())
        pad  = 10
        bbox = (max(0,x1-pad), max(0,y1-pad),
                min(IMG_SIZE-1,x2+pad), min(IMG_SIZE-1,y2+pad))
    else:
        bbox = (0, 0, IMG_SIZE-1, IMG_SIZE-1)

    ulcer_area_pct = float(mask.sum()) / (IMG_SIZE * IMG_SIZE) * 100

    if ulcer_area_pct < 5:
        severity_label, severity_score = 'Mild', 1
    elif ulcer_area_pct < 15:
        severity_label, severity_score = 'Moderate', 2
    else:
        severity_label, severity_score = 'Severe', 3

    # Encode overlay as base64
    overlay_pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_pil.save(buf, format='PNG')
    overlay_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        'predicted_class' : CLASS_NAMES[pred_idx],
        'confidence'      : round(confidence, 4),
        'class_probs'     : class_probs,
        'heatmap_b64'     : overlay_b64,
        'ulcer_bbox'      : bbox,
        'ulcer_area_pct'  : round(ulcer_area_pct, 2),
        'severity_label'  : severity_label,
        'severity_score'  : severity_score,
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
