import os
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

print(" Step 1 complete: Images and landmarks resized and saved.")
