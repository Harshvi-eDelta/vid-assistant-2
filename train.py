'''import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from model import LandmarkCNN
from tqdm import tqdm  # Progress bar

class LandmarkDataset(Dataset):
    def __init__(self, image_dir, t7_dir, transform=None):
        self.image_dir = image_dir
        self.t7_dir = t7_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        t7_path = os.path.join(self.t7_dir, img_name.replace('.jpg', '.t7').replace('.png', '.t7'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        landmarks = torch.load(t7_path)  # (N, 2)
        landmarks = landmarks / 256.0    # Normalize to [0,1]
        landmarks = landmarks.view(-1)   # Flatten to [N*2]

        return image, landmarks

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = LandmarkDataset('/Users/edelta076/Desktop/Project_VID_Assistant2/resized_images/', '/Users/edelta076/Desktop/Project_VID_Assistant2/resized_t7/', transform)
import matplotlib.pyplot as plt

for i in range(10):
    _, landmark = dataset[i]
    lm = landmark.view(-1, 2).numpy() * 256  # de-normalize
    plt.scatter(lm[:, 0], lm[:, 1])
    plt.gca().invert_yaxis()
    plt.title(f"Landmarks sample {i}")
    plt.show()

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LandmarkCNN(num_landmarks=68)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop with Progress Bar
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for images, landmarks in progress_bar:
        images = images.to(device)
        landmarks = landmarks.to(device)

        preds = model(images)
        loss = criterion(preds, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        

    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {epoch_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "landmark_model_1.pth")
print(" Training complete! Model saved as 'landmark_model_1.pth'")'''

# import torch

# # Change this to any .t7 file from your training dataset
# t7_path = "/Users/edelta076/Desktop/Project_VID_Assistant2/resized_t7/1.t7"

# # Load landmarks
# landmarks = torch.load(t7_path)  # shape: (68, 2)

# # Print max X and Y values
# print("Max X:", landmarks[:, 0].max().item())
# print("Max Y:", landmarks[:, 1].max().item())

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from model import LandmarkCNN
from tqdm import tqdm  # Progress bar
from data import get_valid_image_list, FilteredLandmarkDataset

class LandmarkDataset(Dataset):
    def __init__(self, image_dir, t7_dir, transform=None):
        self.image_dir = image_dir
        self.t7_dir = t7_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        t7_path = os.path.join(self.t7_dir, img_name.replace('.jpg', '.t7').replace('.png', '.t7'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        landmarks = torch.load(t7_path)  # (N, 2)
        #landmarks = landmarks / 256.0    # Normalize to [0,1]
        landmarks = landmarks.view(-1)   # Flatten to [N*2]

        return image, landmarks

# Setup
image_dir = "/Users/edelta076/Desktop/Project_VID_Assistant2/resized_images/"
t7_dir = "/Users/edelta076/Desktop/Project_VID_Assistant2/resized_t7/"
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Filter data
valid_list = get_valid_image_list(image_dir, t7_dir)
dataset = FilteredLandmarkDataset(image_dir, t7_dir, valid_list, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LandmarkCNN(num_landmarks=68)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop with Progress Bar
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for images, landmarks in progress_bar:
        images = images.to(device)
        landmarks = landmarks.to(device)

        preds = model(images)
        loss = criterion(preds, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        

    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {epoch_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "landmark_model_1.pth")
print(" Training complete! Model saved as 'landmark_model_1.pth'")

