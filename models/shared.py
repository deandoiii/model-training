import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = [
    'Glass_Bottle',
    'HDPE_Bottle',
    'PET_ClearBottle',
    'PET_With_Cap',
    'PET_With_Liquid',
    'REJECT'
]
NUM_CLASSES = len(CLASSES)

CLASS_DECISION = {
    'Glass_Bottle'    : ('REJECT', 'Non-PET material (Glass)'),
    'HDPE_Bottle'     : ('REJECT', 'Non-PET material (HDPE)'),
    'PET_ClearBottle' : ('ACCEPT', 'Valid PET bottle'),
    'PET_With_Cap'    : ('REJECT', 'Has cap'),
    'PET_With_Liquid' : ('REJECT', 'Has liquid'),
    'REJECT'          : ('REJECT', 'General rejection'),
}

COUNTS        = [118, 51, 197, 60, 71, 134]
TOTAL         = sum(COUNTS)
CLASS_WEIGHTS = torch.tensor(
    [TOTAL / (NUM_CLASSES * c) for c in COUNTS],
    dtype=torch.float32
)

def get_transforms(split='train', img_size=224):
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=20,
                                 val_shift_limit=20, p=0.4),
            A.GaussNoise(var_limit=(5, 30), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomShadow(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class PETDataset(Dataset):
    """
    Reads Roboflow object detection export:
      root/images/img.jpg
      root/labels/img.txt  ← first number is class index
    """
    def __init__(self, root_dir, split='train', img_size=224):
        self.img_dir   = os.path.join(root_dir, 'images')
        self.lbl_dir   = os.path.join(root_dir, 'labels')
        self.transform = get_transforms(split, img_size)

        self.samples = []
        skipped = 0
        for fname in os.listdir(self.img_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            lbl_file = os.path.join(
                self.lbl_dir,
                os.path.splitext(fname)[0] + '.txt'
            )
            if not os.path.exists(lbl_file):
                skipped += 1
                continue
            with open(lbl_file, 'r') as f:
                content = f.read().strip()
            if not content:
                skipped += 1
                continue
            self.samples.append((
                os.path.join(self.img_dir, fname),
                lbl_file
            ))

        print(f'[Dataset] {split}: {len(self.samples)} images loaded, '
              f'{skipped} skipped (empty/missing labels)')

        if len(self.samples) == 0:
            raise RuntimeError(
                f'No valid images found in {self.img_dir}. '
                f'Check your dataset path.'
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        img = np.array(Image.open(img_path).convert('RGB'))
        img = self.transform(image=img)['image']

        label = -1
        with open(lbl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) > 0:
                        label = int(parts[0])
                        break

        if label == -1:
            raise ValueError(f'Could not read label from {lbl_path}')

        return img, label

def get_dataloaders(dataset_dir, batch_size=32, img_size=224):
    train_ds = PETDataset(os.path.join(dataset_dir, 'train'), 'train', img_size)
    val_ds   = PETDataset(os.path.join(dataset_dir, 'valid'), 'val',   img_size)
    test_ds  = PETDataset(os.path.join(dataset_dir, 'test'),  'val',   img_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl, test_dl