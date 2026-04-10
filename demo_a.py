# demo_webcam.py  ← place in root folder
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys, os
sys.path.append('.')

from models.approach_a import PETDetectorA
from models.shared import CLASSES, CLASS_DECISION

# ==========================================
# 1. MODEL SETUP
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Loading model on {DEVICE}...')

model = PETDetectorA(freeze_backbone=False).to(DEVICE)
model.load_state_dict(torch.load('weights/best_a.pt', map_location=DEVICE))
model.eval()
print('Model ready.')

TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def classify(frame_bgr):
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(image=rgb)['image']
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
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

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error: Could not read frame.')
        break

    result = classify(frame)

    # ==========================================
    # 3. DECISION LOGIC
    # ==========================================
    accepted  = result['decision'] == 'ACCEPT'
    cls       = result['class']
    conf      = result['confidence']
    reason    = result['reason']

    # ==========================================
    # 4. UI OVERLAY
    # ==========================================
    # Top banner — green for accept, red for reject
    color = (0, 200, 0) if accepted else (0, 0, 220)
    cv2.rectangle(frame, (0, 0), (640, 60), color, -1)

    decision_text = 'SYSTEM DECISION: ACCEPT' if accepted \
                    else f'SYSTEM DECISION: REJECT'
    cv2.putText(frame, decision_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Class + confidence below banner
    cv2.putText(frame, f'{cls}  ({conf:.0%})', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    # Reject reason
    if not accepted:
        cv2.putText(frame, f'Reason: {reason}', (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 150, 255), 2)

    # All class probabilities on the right side
    y = 80
    for cls_name, prob in result['all_probs'].items():
        bar_w = int(prob * 140)
        bar_color = (0, 180, 0) if cls_name == 'PET_ClearBottle' else (60, 60, 180)
        cv2.rectangle(frame, (480, y-12), (480 + bar_w, y - 2), bar_color, -1)
        cv2.putText(frame, f'{cls_name[:12]}: {prob:.2f}',
                    (480, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (210, 210, 210), 1)
        y += 20

    cv2.imshow('PAR — PET Bottle Classifier', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()