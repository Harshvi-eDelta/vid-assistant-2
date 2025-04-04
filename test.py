import torch
import cv2
import numpy as np
import dlib
from model import LandmarkCNN
from torchvision import transforms

# Initialize Dlib's face detector
face_detector = dlib.get_frontal_face_detector()

# Load trained landmark detection model
model = LandmarkCNN(num_landmarks=68)
model.load_state_dict(torch.load("landmark_model_1.pth", map_location=torch.device('cpu')))
model.eval()

# Load and prepare input image
img_path = "/Users/edelta076/Desktop/Project_VID_Assistant2/face_images/fimg9.jpg"
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {img_path}")
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_detector(gray)

if len(faces) == 0:
    print("⚠️ No face detected!")
    exit()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Loop over all detected faces
for rect in faces:
    x, y, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    w, h = x2 - x, y2 - y

    # Crop and preprocess face
    face_crop = image[y:y+h, x:x+w]
    if face_crop.size == 0:
        continue

    face_tensor = transform(face_crop).unsqueeze(0)

    # Predict normalized landmarks from model
    with torch.no_grad():
        preds = model(face_tensor).cpu().numpy().reshape(-1, 2)

    # Step 1: Convert to 256x256 pixel coordinates
    preds[:, 0] *= 256
    preds[:, 1] *= 256

    # Step 2: Scale back to original face crop size
    scale_x = w / 256.0
    scale_y = h / 256.0
    preds[:, 0] *= scale_x
    preds[:, 1] *= scale_y

    # Step 3: Translate to image coordinates (add top-left corner of face box)
    preds[:, 0] += x
    preds[:, 1] += y

    # Draw face box
    cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 0), 2)

    # Draw landmarks
    for (lx, ly) in preds.astype(int):
        cv2.circle(image, (lx, ly), 2, (0, 255, 0), -1)

# Show result
cv2.imshow("Dlib Face + Landmark Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
