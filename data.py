'''import os
import torch
import cv2
import numpy as np
import torchfile

input_img_dir = '/Users/edelta076/Desktop/Project_VID_Assistant2/original_jpg'
input_t7_dir = '/Users/edelta076/Desktop/Project_VID_Assistant2/t7'

output_img_dir = 'resized_images/'
output_t7_dir = 'resized_t7/'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_t7_dir, exist_ok=True)

for filename in os.listdir(input_img_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load original image and t7 file
        img_path = os.path.join(input_img_dir, filename)
        t7_path = os.path.join(input_t7_dir, filename.replace('.jpg', '.t7').replace('.png', '.t7'))

        image = cv2.imread(img_path)
        orig_h, orig_w = image.shape[:2]
        resized_image = cv2.resize(image, (256, 256))

        landmarks = torchfile.load(t7_path)  # Load as numpy array or dict
        print(f"Landmarks for {filename}: {landmarks[:5]}")
        landmarks = landmarks / 256.0  # ✅ Normalize here to [0, 1] for model training

        #print(landmarks)  # Check format

        landmarks = torch.tensor(landmarks, dtype=torch.float32)  # Convert to PyTorch tensor
        resized_landmarks = landmarks.clone()  # Now .clone() will work

        # Scale landmarks
        scale_x = 256 / orig_w
        scale_y = 256 / orig_h
        resized_landmarks = landmarks.clone()
        resized_landmarks[:, 0] *= scale_x
        resized_landmarks[:, 1] *= scale_y

        # Save resized image and t7
        out_img_path = os.path.join(output_img_dir, filename)
        out_t7_path = os.path.join(output_t7_dir, filename.replace('.jpg', '.t7').replace('.png', '.t7'))

        cv2.imwrite(out_img_path, resized_image)
        torch.save(resized_landmarks, out_t7_path)

print(" Step 1 complete: Images and landmarks resized and saved.")'''

# data.py

import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

def get_valid_image_list(image_dir, t7_dir):
    valid_list = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    for fname in image_files:
        t7_path = os.path.join(t7_dir, fname.replace('.jpg', '.t7').replace('.png', '.t7'))
        if not os.path.exists(t7_path):
            continue
        try:
            landmarks = torch.load(t7_path).numpy()
            if landmarks.shape == (68, 2) and np.all((landmarks >= 0) & (landmarks <= 1)):
                valid_list.append(fname)
        except:
            continue
    return valid_list

class FilteredLandmarkDataset(Dataset):
    def __init__(self, image_dir, t7_dir, valid_list, transform=None):
        self.image_dir = image_dir
        self.t7_dir = t7_dir
        self.valid_list = valid_list
        self.transform = transform

    def __len__(self):
        return len(self.valid_list)

    def __getitem__(self, idx):
        fname = self.valid_list[idx]
        img_path = os.path.join(self.image_dir, fname)
        t7_path = os.path.join(self.t7_dir, fname.replace('.jpg', '.t7').replace('.png', '.t7'))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        landmarks = torch.load(t7_path)  # shape: (68, 2)
        landmarks = landmarks.view(-1)   # Flatten to [136]
        return image, landmarks

