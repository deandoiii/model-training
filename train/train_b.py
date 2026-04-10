import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report
from models.approach_b import BottleClassifier
from models.shared import get_dataloaders, CLASS_WEIGHTS, CLASSES

# ==========================================
# CONFIG
# ==========================================
CFG = {
    'dataset_dir' : 'dataset',
    'batch_size'  : 16,
    'img_size'    : 224,
    'epochs'      : 25,
    'lr'          : 2e-4,
    'patience'    : 7,
    'device'      : 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f'Using device: {CFG["device"]}')

train_dl, val_dl, _ = get_dataloaders(
    CFG['dataset_dir'], CFG['batch_size'], CFG['img_size']
)

model   = BottleClassifier(pretrained=True).to(CFG['device'])
loss_fn = torch.nn.CrossEntropyLoss(
    weight=CLASS_WEIGHTS.to(CFG['device'])
)

# Discriminative learning rates:
# pretrained backbone gets 10x lower LR to preserve ImageNet weights
# new custom head gets full LR
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': CFG['lr'] * 0.1},
    {'params': model.head.parameters(),     'lr': CFG['lr']},
])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[CFG['lr'] * 0.1, CFG['lr']],
    epochs=CFG['epochs'],
    steps_per_epoch=len(train_dl)
)

best_f1    = 0.0
no_improve = 0

for epoch in range(CFG['epochs']):

    # ── Train ──────────────────────────────────────────────
    model.train()
    train_loss = 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(CFG['device']), labels.to(CFG['device'])
        optimizer.zero_grad()
        loss = loss_fn(model(imgs), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

    # ── Validate ───────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_dl:
            preds = model(imgs.to(CFG['device'])).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    val_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = train_loss / len(train_dl)
    print(f'Epoch {epoch+1:02d}/{CFG["epochs"]} | '
          f'loss={avg_loss:.4f} | val_f1={val_f1:.4f}')

    if val_f1 > best_f1:
        best_f1    = val_f1
        no_improve = 0
        os.makedirs('weights', exist_ok=True)
        torch.save(model.state_dict(), 'weights/best_b.pt')
        print(f'  ✓ Saved best model (F1={best_f1:.4f})')
    else:
        no_improve += 1
        if no_improve >= CFG['patience']:
            print(f'Early stopping at epoch {epoch+1}')
            break

# ── Final report ───────────────────────────────────────────
print('\nLoading best model for final report...')
model.load_state_dict(torch.load('weights/best_b.pt', map_location=CFG['device']))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_dl:
        preds = model(imgs.to(CFG['device'])).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print('\nFinal Validation Report:')
print(classification_report(all_labels, all_preds,
                             target_names=CLASSES, zero_division=0))
print(f'Best Val F1: {best_f1:.4f}')
print('Weights saved to weights/best_b.pt')