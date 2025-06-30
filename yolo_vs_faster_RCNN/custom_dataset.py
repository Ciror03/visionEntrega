import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class YoloToFRCNNDataset(Dataset):
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".JPG", ".png"))
        ])
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(image_name)[0] + ".txt")

        # Cargar imagen
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Leer anotaciones YOLO
        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f.readlines():
                cls, xc, yc, w, h = map(float, line.strip().split())
                # Convertir a formato x1, y1, x2, y2 en píxeles
                x1 = (xc - w / 2) * width
                y1 = (yc - h / 2) * height
                x2 = (xc + w / 2) * width
                y2 = (yc + h / 2) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)  # PyTorch espera etiquetas desde 1

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
    
