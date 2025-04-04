import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkCNN(nn.Module):
    def __init__(self, num_landmarks=68):
        super(LandmarkCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_landmarks * 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 128x128
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 64x64
        x = F.relu(F.max_pool2d(self.conv3(x), 2))  # 32x32
        x = x.view(x.size(0), -1)                   # flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))              # Normalized landmarks [0, 1]
        return x
