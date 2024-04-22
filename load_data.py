import os 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Define path
data_dir = f"/workspace/geopacha/data/region1_data/train"

# 
num_workers = 1 # TODO - check
batch_size = 512

device = torch.device("cuda")

data_transforms = transforms.Compose([
            transforms.Resize(size=196, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# def resize_image(image, patch_height, patch_width):
#     width, height = image.size()
#     new_width = (width // patch_width) * patch_width
#     new_height = (height // patch_height) * patch_height
#     resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
#     return resized_image

class GeoPACHAImageDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.image_paths = pd.read_csv(data_file, delimiter=",")
        self.transform = transform
        # TODO target_transform = None ? 

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(data_dir, self.image_paths['img_path'][idx])
        image = np.load(img_path)
        image = image[:3]
        image = np.transpose(image, (1, 2, 0))  # Now shape becomes (256, 256, 3)
        # Step 3: Convert to 'uint8' if it's not already
        # if image.dtype != np.uint8:
        #     # Normalize (if needed) and convert
        #     image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
        #     image = (image * 255).astype(np.uint8)  # Scale to [0, 255]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        # patch_height = 14
        # patch_width = 14
        # image = resize_image(image, patch_height, patch_width)
        label = self.image_paths['label'][idx]
        
        return image, label