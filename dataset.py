from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import Dataset
import numpy as np
import torch
train_transform = A.Compose([
    A.Resize(384,384),
    A.RandomCrop(320,320),
    A.HorizontalFlip(p=0.5),
    A.Affine(
        translate_percent=(-0.1, 0.1),
        scale=(0.8, 1.2),
        rotate=(-30, 30),
        p=0.7
    ),
    A.OneOf([
        A.MotionBlur(),
        A.GaussianBlur(),
        A.GaussNoise()
    ], p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(),
        A.HueSaturationValue()
    ], p=0.5),
    A.Normalize(),
    ToTensorV2()
])


val_transform = A.Compose([
    A.Resize(320,320),
    A.Normalize(),
    ToTensorV2()  
])


class PetDataset(Dataset):
    def __init__(self, torchvision_dataset, transform=None):
        self.dataset = torchvision_dataset
        self.transform = transform
    def __len__(self):
       return len(self.dataset) 
    def __getitem__(self,idx):
        image, mask = self.dataset[idx]
        image = np.array(image.convert("RGB"))
        mask = np.array(mask)
        mask = mask -1
        if self.transform:
            augmented =self.transform(image=image, mask =mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask.long()


