# save as check_dataset.py in root folder, run once then delete
import sys, os
sys.path.append('.')
from models.shared import get_dataloaders

if __name__ == '__main__':
    train_dl, val_dl, test_dl = get_dataloaders('dataset', batch_size=4)

    # Check one batch
    imgs, labels = next(iter(train_dl))
    print(f'Image batch shape : {imgs.shape}')   # should be [4, 3, 224, 224]
    print(f'Label batch       : {labels}')        # should be tensor of 0-5 values
    print('Dataset check passed!')