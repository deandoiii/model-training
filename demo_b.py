# demo_webcam_b.py  ← place in root folder
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
import sys, os
sys.path.append('.')

from models.approach_b import BottleClassifier
from models.shared import CLASSES, CLASS_DECISION

# ==========================================
# 1. MODEL SETUP
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Loading models on {DEVICE}...')

# Stage 1 — YOLO detector
detector = YOLO('weights/yolo_detector.pt')

# Stage 2 — EfficientNet classifier
classifier = BottleClassifier(pretrained=False).to(DEVICE)
classifier.load_state_dict(
    torch.load('weights/best_b_effnet.pt', map_location=DEVICE)
)
classifier.eval()
print('Both models ready.')

TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def classify_crop(crop_rgb):
    """Run EfficientNet on a cropped bottle region."""
    tensor = TRANSFORM(image=crop_rgb)['image']
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(classifier(tensor), dim=1)[0]
    idx              = probs.argmax().item()
    class_name       = CLASSES[idx]
    decision, reason = CLASS_DECISION[class_name]
    return {
        'class'     : class_name,
        'confidence': float(probs[idx]),
        'decision'  : decision,
        'reason'    : reason,
        'all_probs' : {c: float(probs[i]) for i, c in enumerate(CLASSES)}
    }

# ==========================================
# 2. WEBCAM SETUP
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print('Webcam ready. Press Q to quit.')

frame_count = 0
last_results = []   # cache last detection so display stays smooth

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error: Could not read frame.')
        break

    frame_count += 1

    # ==========================================
    # 2. DECISION LOGIC — run every 3rd frame
    # ==========================================
    if frame_count % 3 == 0:
        detections = detector(frame, conf=0.4, verbose=False)[0]
        last_results = []

        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            h, w = frame.shape[:2]
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(w, x2 + 10)
            y2 = min(h, y2 + 10)

            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            result = classify_crop(crop)
            result['bbox'] = (x1, y1, x2, y2)
            last_results.append(result)

    # ==========================================
    # 3. UI OVERLAY
    # ==========================================
    if not last_results:
        # No bottle detected
        cv2.rectangle(frame, (0, 0), (640, 55), (50, 50, 50), -1)
        cv2.putText(frame, 'No bottle detected', (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
    else:
        for r in last_results:
            x1, y1, x2, y2 = r['bbox']
            accepted = r['decision'] == 'ACCEPT'
            color    = (0, 200, 0) if accepted else (0, 0, 220)

            # Bounding box around detected bottle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label above bounding box
            box_label = f"{r['class']} {r['confidence']:.0%}"
            cv2.putText(frame, box_label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 2)

        # Use first detection for top banner
        main = last_results[0]
        accepted = main['decision'] == 'ACCEPT'
        color    = (0, 200, 0) if accepted else (0, 0, 220)

        cv2.rectangle(frame, (0, 0), (640, 55), color, -1)
        decision_text = 'SYSTEM DECISION: ACCEPT' if accepted \
                        else 'SYSTEM DECISION: REJECT'
        cv2.putText(frame, decision_text, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Class + reason below banner
        cv2.putText(frame, f"{main['class']}  ({main['confidence']:.0%})",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (200, 200, 200), 2)
        if not accepted:
            cv2.putText(frame, f"Reason: {main['reason']}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 150, 255), 2)

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

    # Stage label bottom left
    cv2.putText(frame, 'Approach B: YOLOv11 + EfficientNet',
                (10, 468), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (120, 120, 120), 1)

    cv2.imshow('PAR — Approach B: PET Bottle Classifier', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()