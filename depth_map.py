import os
import cv2
import torch
import numpy as np
from torchvision.transforms import Compose
from tqdm import tqdm

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Paths
input_folder = "/Users/edelta076/Desktop/Project_VID_Assistant2/original_jpg"
output_folder = "/Users/edelta076/Desktop/Project_VID_Assistant2/depth_maps"
os.makedirs(output_folder, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Process each image
for img_name in tqdm(os.listdir(input_folder)):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess
        input_tensor = transform(img_rgb).to(device)

        # Inference
        with torch.no_grad():
            prediction = midas(input_tensor)[0]
            prediction = prediction.cpu().numpy()

        # Normalize depth map for saving
        depth_min = prediction.min()
        depth_max = prediction.max()
        depth_vis = 255 * (prediction - depth_min) / (depth_max - depth_min)
        depth_vis = depth_vis.astype("uint8")

        # Save result
        output_path = os.path.join(output_folder, img_name.split('.')[0] + "_depth.png")
        cv2.imwrite(output_path, depth_vis)

# Debug one image only
'''for img_name in os.listdir(input_folder):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(f"Original: {img_rgb.shape}")

        input_tensor = transform(img_rgb).to(device)
        print(f"Transformed: {input_tensor.shape}")

        break  # ✅ Stop after first image'''
