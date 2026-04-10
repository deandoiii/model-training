import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from models.shared import CLASSES, CLASS_DECISION, NUM_CLASSES


# ==========================================
# STAGE 2: EfficientNet-B0 Classifier
# ==========================================
class BottleClassifier(nn.Module):
    """
    EfficientNet-B0 backbone with custom classification head.
    B0 chosen over B3 because dataset is small — less overfitting risk.
    Input: 224x224 cropped bottle image
    Output: class logits (6 classes)
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0       # remove original head
        )
        in_features = self.backbone.num_features  # 1280 for B0
        print(f'[Model B] EfficientNet in_features: {in_features}')

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ==========================================
# INFERENCE TRANSFORM (no augmentation)
# ==========================================
CLASSIFY_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


# ==========================================
# FULL TWO-STAGE PIPELINE (for inference only)
# ==========================================
class PETPipelineB:
    """
    Combines YOLO detector + EfficientNet classifier.
    Used at inference time, not during training.
    """
    def __init__(self, detector_weights, classifier_weights, device='cpu'):
        # Stage 1: YOLO detector
        self.detector = YOLO(detector_weights)

        # Stage 2: EfficientNet classifier
        self.classifier = BottleClassifier(pretrained=False).to(device)
        self.classifier.load_state_dict(
            torch.load(classifier_weights, map_location=device)
        )
        self.classifier.eval()
        self.device = device
        print('Pipeline B ready.')

    def run(self, frame_bgr, conf=0.4):
        results    = []
        detections = self.detector(frame_bgr, conf=conf, verbose=False)[0]

        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            # Add padding around crop
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(frame_bgr.shape[1], x2 + 10)
            y2 = min(frame_bgr.shape[0], y2 + 10)

            crop   = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            tensor = CLASSIFY_TRANSFORM(image=crop)['image'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                probs = torch.softmax(self.classifier(tensor), dim=1)[0]

            idx              = probs.argmax().item()
            class_name       = CLASSES[idx]
            decision, reason = CLASS_DECISION[class_name]

            results.append({
                'bbox'      : (x1, y1, x2, y2),
                'class'     : class_name,
                'confidence': float(probs[idx]),
                'decision'  : decision,
                'reason'    : reason,
                'all_probs' : {c: float(probs[i]) for i, c in enumerate(CLASSES)}
            })
        return results


def predict_b_single(classifier, image_tensor, device='cpu'):
    """Classify a single pre-cropped image tensor. Used during evaluation."""
    classifier.eval()
    with torch.no_grad():
        logits = classifier(image_tensor.unsqueeze(0).to(device))
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = probs.argmax().item()
    class_name       = CLASSES[idx]
    decision, reason = CLASS_DECISION[class_name]
    return {
        'class'     : class_name,
        'confidence': float(probs[idx]),
        'decision'  : decision,
        'reason'    : reason,
        'all_probs' : {c: float(probs[i]) for i, c in enumerate(CLASSES)}
    }