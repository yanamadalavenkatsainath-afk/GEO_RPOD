"""SPEED+ dataset loader for pose-CNN calibration training."""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224


def _repeat_to_3ch(x):
    """1->3 channels so the grayscale image fits a pretrained RGB backbone."""
    return x.repeat(3, 1, 1)


_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                            # (1,H,W) in [0,1], grayscale
    transforms.Lambda(_repeat_to_3ch),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SpeedPlusDataset(Dataset):
    """Loads (image, position[m], quaternion[wxyz]) triples from a SPEED+ split."""

    def __init__(self, root, split_json, images_subdir="images", transform=None, limit=None):
        self.images_dir = os.path.join(root, images_subdir)
        with open(os.path.join(root, split_json)) as f:
            self.labels = json.load(f)
        if limit is not None:
            self.labels = self.labels[:limit]
        self.transform = transform or _TRANSFORM

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels[idx]
        img = Image.open(os.path.join(self.images_dir, item["filename"])).convert("L")
        img = self.transform(img)
        pos = torch.tensor(item["r_Vo2To_vbs_true"], dtype=torch.float32)
        quat = torch.tensor(item["q_vbs2tango_true"], dtype=torch.float32)
        quat = quat / quat.norm()
        return img, pos, quat
