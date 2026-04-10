# evaluate/evaluate.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from models.shared import get_dataloaders, CLASSES, CLASS_DECISION

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_evaluation(model, test_dl, model_name='model'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_dl:
            logits = model(imgs.to(DEVICE))
            preds  = torch.softmax(logits, dim=1).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Report ─────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  EVALUATION — {model_name}')
    print(f'{"="*60}')
    print(f'Accuracy : {accuracy_score(all_labels, all_preds):.4f}')
    print()
    print(classification_report(all_labels, all_preds,
                                 target_names=CLASSES, zero_division=0))

    # ── Accept/Reject accuracy ─────────────────────────────
    true_dec = [CLASS_DECISION[CLASSES[l]][0] for l in all_labels]
    pred_dec = [CLASS_DECISION[CLASSES[p]][0] for p in all_preds]
    correct  = sum(t == p for t, p in zip(true_dec, pred_dec))
    print(f'Accept/Reject Accuracy: {correct}/{len(all_labels)} '
          f'= {correct/len(all_labels):.4f}')

    # ── Confusion matrix ───────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs('evaluate', exist_ok=True)
    out_path = f'evaluate/confusion_{model_name}.png'
    plt.savefig(out_path, dpi=120)
    plt.show()
    print(f'Confusion matrix saved to {out_path}')


if __name__ == '__main__':
    _, _, test_dl = get_dataloaders('dataset', batch_size=16)

    # ── Evaluate Approach A ────────────────────────────────
    from models.approach_a import PETDetectorA
    model_a = PETDetectorA(freeze_backbone=False).to(DEVICE)
    model_a.load_state_dict(torch.load('weights/best_a.pt', map_location=DEVICE))
    run_evaluation(model_a, test_dl, 'Approach_A')

    # ── Evaluate Approach B ────────────────────────────────
    from models.approach_b import BottleClassifier
    model_b = BottleClassifier(pretrained=False).to(DEVICE)
    model_b.load_state_dict(torch.load('weights/best_b.pt', map_location=DEVICE))
    run_evaluation(model_b, test_dl, 'Approach_B')