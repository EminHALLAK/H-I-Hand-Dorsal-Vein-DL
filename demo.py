import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.decoder(x)
        return x

def load_and_preprocess_image(path, size=(284, 284)):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.resize(image, size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)

class VeinDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_and_preprocess_image(self.image_paths[idx])
        mask = load_and_preprocess_image(self.mask_paths[idx])
        return image, mask

# Correct paths to mask images
image_paths = ['images/org/1.bmp', 'images/org/3.bmp']
mask_paths = ['images/masked/1.bmp', 'images/masked/3.bmp']

# Ensure the paths are correct
for path in image_paths + mask_paths:
    if not os.path.exists(path):
        print(f"File not found: {path}")

dataset = VeinDataset(image_paths, mask_paths)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
for epoch in range(10):
    for images, masks in dataloader:
        output = model(images)
        loss = criterion(output, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

def preprocess_with_model(image_path, model):
    image = load_and_preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
    output_np = output.squeeze().numpy()
    output_np = (output_np > 0.5).astype(np.uint8) * 255  # Binarize and convert to uint8
    return output_np

# Apply preprocessing to a new image
model.eval()
processed_image = preprocess_with_model("path_to_new_image", model)
cv2.imwrite("processed_image.png", processed_image)

def refine_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

refined_mask = refine_mask(processed_image)
cv2.imwrite("refined_image.png", refined_mask)