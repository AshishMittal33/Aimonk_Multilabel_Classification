import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class MultiLabelDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        self.image_names = []
        self.labels = []

        with open(label_file, "r") as f:
            lines = f.readlines()

        # Skip header if exists
        if "Image" in lines[0]:
            lines = lines[1:]

        for line in lines:
            parts = line.strip().split()

            img_name = parts[0]
            img_path = os.path.join(image_dir, img_name)

            # Skip if image file does not exist
            if not os.path.exists(img_path):
                continue

            attrs = []

            for value in parts[1:]:
                if value == "NA":
                    attrs.append(-1)
                else:
                    attrs.append(int(value))

            self.image_names.append(img_name)
            self.labels.append(attrs)

        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
