# demo_webcam_b.py — runs on both laptop (PyTorch) and Pi (NCNN)
import cv2
import numpy as np
import time
import platform
import sys
import os
sys.path.append('.')

# ── Platform detection ────────────────────────────────────
IS_PI  = platform.machine() in ['aarch64', 'armv7l']
DEVICE = 'cpu'
print(f'Platform: {platform.machine()} → {"Raspberry Pi" if IS_PI else "Laptop/Desktop"}')

# ── Shared constants ──────────────────────────────────────
CLASSES = [
    'Glass_Bottle', 'HDPE_Bottle', 'PET_ClearBottle',
    'PET_With_Cap', 'PET_With_Liquid', 'REJECT'
]
CLASS_DECISION = {
    'Glass_Bottle'    : ('REJECT', 'Non-PET (Glass)'),
    'HDPE_Bottle'     : ('REJECT', 'Non-PET (HDPE)'),
    'PET_ClearBottle' : ('ACCEPT', 'Valid PET'),
    'PET_With_Cap'    : ('REJECT', 'Has cap'),
    'PET_With_Liquid' : ('REJECT', 'Has liquid'),
    'REJECT'          : ('REJECT', 'General rejection'),
}
MEAN        = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD         = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONF_THRESH = 0.3

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

# ==========================================
# 1. MODEL LOADING — auto switches per platform
# ==========================================

if IS_PI:
    # ── Pi: NCNN for both models ──────────────────────────
    import ncnn

    net_det = ncnn.Net()
    net_det.opt.use_vulkan_compute = False
    net_det.opt.num_threads        = 4
    net_det.load_param('weights/yolo_detector.param')
    net_det.load_model('weights/yolo_detector.bin')
    print('✅ YOLO detector loaded (NCNN)')

    net_cls = ncnn.Net()
    net_cls.opt.use_vulkan_compute = False
    net_cls.opt.num_threads        = 4
    net_cls.load_param('weights/effnet_b.param')
    net_cls.load_model('weights/effnet_b.bin')
    print('✅ EfficientNet classifier loaded (NCNN)')

    def _detect(frame_bgr):
        """YOLO NCNN — returns list of (x1, y1, x2, y2, conf)."""
        h, w = frame_bgr.shape[:2]
        img  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        img  = img.transpose(2, 0, 1)

        ex = net_det.create_extractor()
        ex.input('image', ncnn.Mat(img))
        _, mat_out = ex.extract('output0')

        preds = np.array(mat_out)
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] < preds.shape[-1]:
            preds = preds.T

        boxes = []
        for pred in preds:
            conf = float(pred[4:].max())
            if conf < CONF_THRESH:
                continue
            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
            x1 = max(0, int((cx - bw / 2) / 640 * w) - 10)
            y1 = max(0, int((cy - bh / 2) / 640 * h) - 10)
            x2 = min(w, int((cx + bw / 2) / 640 * w) + 10)
            y2 = min(h, int((cy + bh / 2) / 640 * h) + 10)
            boxes.append((x1, y1, x2, y2, conf))
        return boxes

    def _classify(crop_bgr):
        """EfficientNet NCNN — returns class + decision."""
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(np.float32)
        img = (img / 255.0 - MEAN) / STD
        img = img.transpose(2, 0, 1)

        ex = net_cls.create_extractor()
        ex.input('image', ncnn.Mat(img))
        _, mat_out = ex.extract('logits')

        probs    = softmax(np.array(mat_out).flatten())
        idx      = probs.argmax()
        cls      = CLASSES[idx]
        decision, reason = CLASS_DECISION[cls]
        return {
            'class'     : cls,
            'confidence': float(probs[idx]),
            'decision'  : decision,
            'reason'    : reason,
            'all_probs' : {c: float(probs[i]) for i, c in enumerate(CLASSES)}
        }

else:
    # ── Laptop: PyTorch for both models ───────────────────
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from ultralytics import YOLO
    from models.approach_b import BottleClassifier

    TRANSFORM = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    _detector = YOLO('weights/yolo_detector.pt')
    print('✅ YOLO detector loaded (PyTorch)')

    _classifier = BottleClassifier(pretrained=False).to(DEVICE)
    _classifier.load_state_dict(
        torch.load('weights/best_b_effnet.pt', map_location=DEVICE)
    )
    _classifier.eval()
    print('✅ EfficientNet classifier loaded (PyTorch)')

    def _detect(frame_bgr):
        """YOLO PyTorch — returns list of (x1, y1, x2, y2, conf)."""
        h, w       = frame_bgr.shape[:2]
        detections = _detector(frame_bgr, conf=CONF_THRESH, verbose=False)[0]
        boxes      = []
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf            = float(box.conf.item())
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(w, x2 + 10)
            y2 = min(h, y2 + 10)
            boxes.append((x1, y1, x2, y2, conf))
        return boxes

    def _classify(crop_bgr):
        """EfficientNet PyTorch — returns class + decision."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor   = TRANSFORM(image=np.array(crop_rgb))['image']
        tensor   = tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(_classifier(tensor), dim=1)[0]
        idx              = probs.argmax().item()
        cls              = CLASSES[idx]
        decision, reason = CLASS_DECISION[cls]
        return {
            'class'     : cls,
            'confidence': float(probs[idx]),
            'decision'  : decision,
            'reason'    : reason,
            'all_probs' : {c: float(probs[i]) for i, c in enumerate(CLASSES)}
        }

# ── Unified classify_frame — same on both platforms ───────
def classify_frame(frame_bgr):
    boxes = _detect(frame_bgr)
    if not boxes:
        return None, []
    results = []
    crops   = []
    for (x1, y1, x2, y2, conf) in boxes:
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        result         = _classify(crop)
        result['bbox'] = (x1, y1, x2, y2)
        results.append(result)
        crops.append(crop.copy())
    return results, crops

# ==========================================
# 2. CAMERA SETUP — auto switches per platform
# ==========================================
USE_PICAMERA = False

if IS_PI:
    try:
        from picamera2 import Picamera2
        cam = Picamera2(0)
        cam.configure(cam.create_video_configuration(
            main={"format": "XRGB8888", "size": (640, 480)}
        ))
        cam.start()
        time.sleep(1)
        USE_PICAMERA = True
        print('📷 Pi Camera started')
    except Exception as e:
        print(f'⚠️  Pi Camera not found ({e}), falling back to USB')

if not USE_PICAMERA:
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cam = cv2.VideoCapture(cam_index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cam.isOpened():
        print('❌ No camera found. Exiting.')
        exit(1)
    print(f'📷 Camera started (index {cam_index})')

print('Running — press Q to quit')
print('Two windows: main feed + crop view')

# ==========================================
# 3. MAIN LOOP
# ==========================================
frame_count  = 0
last_results = []
last_crops   = []

while True:
    # ── Capture ──────────────────────────────────────────
    if USE_PICAMERA:
        frame_bgra = cam.capture_array()
        frame      = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
    else:
        ret, frame = cam.read()
        if not ret:
            continue

    frame_count += 1

    # ── Inference every 3rd frame ─────────────────────────
    if frame_count % 3 == 0:
        t0                    = time.time()
        last_results, last_crops = classify_frame(frame)
        ms                    = (time.time() - t0) * 1000
        if last_results:
            r = last_results[0]
            print(f"[{r['decision']}] {r['class']} "
                  f"({r['confidence']:.2f}) | {ms:.0f}ms")

    # ── Crop window ───────────────────────────────────────
    if last_crops:
        display_crops = []
        for i, crop in enumerate(last_crops):
            c = cv2.resize(crop, (224, 224))
            if i < len(last_results):
                r     = last_results[i]
                color = (0, 200, 0) if r['decision'] == 'ACCEPT' \
                        else (0, 0, 220)
                cv2.putText(c, r['class'], (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.putText(c, f"{r['confidence']:.0%}", (5, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)
                cv2.rectangle(c, (0, 0), (223, 223), color, 4)
            display_crops.append(c)
        crop_canvas = np.hstack(display_crops) \
                      if len(display_crops) > 1 else display_crops[0]
        cv2.imshow('EfficientNet Crop View', crop_canvas)
    else:
        placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'No bottle', (45, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        cv2.imshow('EfficientNet Crop View', placeholder)

    # ── Main feed overlay ─────────────────────────────────
    if not last_results:
        cv2.rectangle(frame, (0, 0), (640, 55), (50, 50, 50), -1)
        cv2.putText(frame, 'No bottle detected', (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
    else:
        for r in last_results:
            x1, y1, x2, y2 = r['bbox']
            accepted = r['decision'] == 'ACCEPT'
            color    = (0, 200, 0) if accepted else (0, 0, 220)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Filled label tag above box
            label       = f"{r['class']}  {r['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            tag_y = max(y1 - 10, th + 6)
            cv2.rectangle(frame,
                          (x1, tag_y - th - 6),
                          (x1 + tw + 8, tag_y + 2),
                          color, -1)
            cv2.putText(frame, label, (x1 + 4, tag_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2)

        # Top banner
        main     = last_results[0]
        accepted = main['decision'] == 'ACCEPT'
        color    = (0, 200, 0) if accepted else (0, 0, 220)
        cv2.rectangle(frame, (0, 0), (640, 55), color, -1)
        cv2.putText(frame,
                    'SYSTEM DECISION: ACCEPT' if accepted
                    else 'SYSTEM DECISION: REJECT',
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)
        cv2.putText(frame,
                    f"{main['class']}  ({main['confidence']:.0%})",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (200, 200, 200), 2)
        if not accepted:
            cv2.putText(frame, f"Reason: {main['reason']}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (80, 150, 255), 2)

        # Probability bars — right side
        y = 80
        for cls_name, prob in main['all_probs'].items():
            bar_w     = int(prob * 140)
            bar_color = (0, 180, 0) if cls_name == 'PET_ClearBottle' \
                        else (80, 60, 180)
            cv2.rectangle(frame, (480, y - 12),
                          (480 + bar_w, y - 2), bar_color, -1)
            cv2.putText(frame, f'{cls_name[:12]}: {prob:.2f}',
                        (480, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (210, 210, 210), 1)
            y += 20

    # Platform label bottom left
    backend = 'NCNN' if IS_PI else 'PyTorch'
    cv2.putText(frame,
                f'Approach B: YOLOv11n + EfficientNet-B0 ({backend})',
                (10, 468), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (120, 120, 120), 1)

    cv2.imshow('PAR — Approach B: PET Bottle Classifier', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────
if USE_PICAMERA:
    cam.stop()
else:
    cam.release()
cv2.destroyAllWindows()