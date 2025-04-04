import os
import cv2
import numpy as np
import scipy.io

# Paths
original_img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant2/300W-3D2/HELEN"
original_mat_dir = "/Users/edelta076/Desktop/Project_VID_Assistant2/300W-3D2/HELEN"
resized_img_dir = "/Users/edelta076/Desktop/Project_VID_Assistant2/resized_images_w/"
resized_mat_dir = "/Users/edelta076/Desktop/Project_VID_Assistant2/resized_mat_w/"

os.makedirs(resized_img_dir, exist_ok=True)
os.makedirs(resized_mat_dir, exist_ok=True)

target_size = (256, 256)

def preprocess():
    for img_name in os.listdir(original_img_dir):
        if not (img_name.endswith(".jpg") or img_name.endswith(".png")):
            continue

        img_path = os.path.join(original_img_dir, img_name)
        mat_path = os.path.join(original_mat_dir, img_name.replace(".jpg", ".mat").replace(".png", ".mat"))

        if not os.path.exists(mat_path):
            print(f" Skipping {img_name}: No .mat file found.")
            continue

        image = cv2.imread(img_path)
        h, w, _ = image.shape
        resized_image = cv2.resize(image, target_size)

        mat_data = scipy.io.loadmat(mat_path)
        landmarks = mat_data['pt2d'].T  # Adjust if your key is different

        # Resize landmarks
        landmarks[:, 0] *= (target_size[0] / w)  
        landmarks[:, 1] *= (target_size[1] / h)  

        # Normalize
        landmarks /= 256.0  

        cv2.imwrite(os.path.join(resized_img_dir, img_name), resized_image)
        scipy.io.savemat(os.path.join(resized_mat_dir, img_name.replace(".jpg", ".mat").replace(".png", ".mat")), {"landmarks": landmarks})

        print(f"Processed {img_name}")

preprocess()
