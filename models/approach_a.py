import torch
import torch.nn as nn
from ultralytics import YOLO
from models.shared import CLASSES, CLASS_DECISION, NUM_CLASSES

class CustomClassHead(nn.Module):
    def __init__(self, in_channels, num_classes=NUM_CLASSES):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(self.pool(x))

class PETDetectorA(nn.Module):
    def __init__(self, yolo_weights='yolo11n.pt', freeze_backbone=True):
        super().__init__()
        yolo = YOLO(yolo_weights)
        self.backbone = yolo.model.model[:10]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Auto-detect output channels
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out   = self.backbone(dummy)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            in_channels = out.shape[1]
        print(f'[Model A] Backbone output channels: {in_channels}')

        self.head = CustomClassHead(in_channels=in_channels)

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        return self.head(features)

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

def predict_a(model, image_tensor, device='cpu'):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
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