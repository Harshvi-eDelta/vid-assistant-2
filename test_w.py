import torch
import cv2
import numpy as np
from model import LandmarkCNN
from torchvision import transforms

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained model
model = LandmarkCNN(num_landmarks=68)
model.load_state_dict(torch.load("landmark_model_w.pth"))
model.eval()

# Load and preprocess image
img_path = "/Users/edelta076/Desktop/Project_VID_Assistant2/face_images/fimg9.jpg"
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w, _ = image.shape

# Detect face
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
if len(faces) == 0:
    print("⚠️ No face detected!")
    exit()

# Process the first detected face
x, y, w, h = faces[0]
face = image[y:y+h, x:x+w]  # Crop face

# Transform face for model input
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
face_tensor = transform(face).unsqueeze(0)

# Predict landmarks
with torch.no_grad():
    preds = model(face_tensor).cpu().numpy().reshape(-1, 2)

print("Raw model output (in 256x256 space):", preds[:5])

# If training was done on absolute pixel coordinates (not normalized)
preds[:, 0] += x
preds[:, 1] += y

print("Landmarks mapped to original image:", preds[:5])

# Draw face box
cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Draw landmarks
for (lx, ly) in preds.astype(int):
    cv2.circle(image, (lx, ly), 1, (0, 255, 0), -1)

# Show result
cv2.imshow("Face & Landmark Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
