import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import scipy.io
from tqdm import tqdm
from model import LandmarkCNN
import scipy.io as sio

class LandmarkDataset(Dataset):
    def __init__(self, image_dir, mat_dir, transform=None):
        self.image_dir = image_dir
        self.mat_dir = mat_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mat_path = os.path.join(self.mat_dir, img_name.replace('.jpg', '.mat').replace('.png', '.mat'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        mat_data = sio.loadmat(mat_path)
        landmarks = mat_data['landmarks']   
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        landmarks = landmarks / 256.0  # This must be done during training
        landmarks = landmarks.reshape(-1)

        return image, landmarks

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = LandmarkDataset('/Users/edelta076/Desktop/Project_VID_Assistant2/resized_images_w/', '/Users/edelta076/Desktop/Project_VID_Assistant2/resized_mat_w/', transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LandmarkCNN(num_landmarks=68)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, landmarks in progress_bar:
        images, landmarks = images.to(device), landmarks.to(device)
        preds = model(images)
        loss = criterion(preds, landmarks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {epoch_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "landmark_model_w.pth")
print(" Training complete! Model saved as 'landmark_model_w.pth'")
