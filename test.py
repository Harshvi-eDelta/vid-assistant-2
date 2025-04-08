'''import torch
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
img_path = "/Users/edelta076/Desktop/Project_VID_Assistant2/face_images/fimg2.jpg"
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

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(256),
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

    preds *= 256

    # Step 1: Convert to 256x256 pixel coordinates
    # preds[:, 0] *= 256
    # preds[:, 1] *= 256

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
cv2.destroyAllWindows()'''

import torch
import cv2
import numpy as np
import dlib
from model import LandmarkCNN
from torchvision import transforms

# Load model
model = LandmarkCNN(num_landmarks=68)
model.load_state_dict(torch.load("landmark_model_1.pth", map_location=torch.device('cpu')))
model.eval()

# Dlib face detector
face_detector = dlib.get_frontal_face_detector()

# Image
img_path = "/Users/edelta076/Desktop/Project_VID_Assistant2/face_images/fimg2.jpg"
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

# Preprocessing transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

for rect in faces:
    x, y, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    w, h = x2 - x, y2 - y

    # Crop face
    face_crop = image[y:y+h, x:x+w]
    if face_crop.size == 0:
        continue

    # Debugging outputs
    print(f"Face coordinates: x={x}, y={y}, x2={x2}, y2={y2}")
    print(f"Crop dimensions (w, h): {w}, {h}")

    # Resize + normalize input
    face_tensor = transform(face_crop).unsqueeze(0)

    # Inference
    with torch.no_grad():
        preds = model(face_tensor).cpu().numpy().reshape(-1, 2)

    # === Landmark Transformation Steps ===
    #print(f"Preds (before scaling): {preds}")

    # 1. Preds are normalized (0-1). Multiply by 256 to match input image size to model
    print(f"Preds (before scaling): {preds[:5]}")
    preds *= 512.0
    print(f"Preds (after scaling): {preds[:5]}")

    # Get the face crop dimensions (w and h)
    # h = y2 - y  # height of the detected face
    # w = x2 - x  # width of the detected face

    # Scale the predicted landmarks from 256x256 to the original image size
    # If preds are in the [0, 1] range for the 256x256 image, scale them to the original image size

    scale_x = w / 256.0  # Scale for width based on the face crop
    scale_y = h / 256.0  # Scale for height based on the face crop

    print(f"Scale factors: scale_x={scale_x}, scale_y={scale_y}")

    # Apply scaling to the predicted landmarks
    preds[:, 0] *= scale_x  # Scale X coordinates
    preds[:, 1] *= scale_y  # Scale Y coordinates

    print(f"Preds (after scaling to crop): {preds[:5]}") 

    # Translate the landmarks back to the original image's coordinate system
    preds[:, 0] += x  # Add the X offset of the face bounding box
    preds[:, 1] += y  # Add the Y offset of the face bounding box

    print(f"Preds (after translating to image space): {preds[:5]}")

    # Draw bounding box and landmarks on the original image for debugging
    # cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 0), 2)  # Face bounding box
    # print(f"Face coordinates: x={x}, y={y}, x2={x2}, y2={y2}")
    # print(f"Width: {w}, Height: {h}")

    for lx, ly in preds.astype(int):
        cv2.circle(image, (lx, ly), 2, (0, 255, 0), -1)  # Draw landmarks
    
    cv2.rectangle(image, (x, y), (x2, y2), (255, 0, 0), 2)  # Face bounding box

    # Print out the values before scaling
    print(f"Face bounding box: x={x}, y={y}, x2={x2}, y2={y2}")
    print(f"Preds before scaling: {preds}")

    # Apply scaling to the predicted landmarks
    preds_scaled = preds * np.array([scale_x, scale_y])

    # Print the scaled values to verify they are correct
    print(f"Preds after scaling: {preds_scaled}")

    # Apply translation to the image coordinates
    preds_translated = preds_scaled + np.array([x, y])
    print(f"Preds after translation to image space: {preds_translated}")

    # Check for landmarks that are outside the image boundaries
    for (x_pred, y_pred) in preds_translated.astype(int):
        if x_pred < 0 or x_pred >= image.shape[1] or y_pred < 0 or y_pred >= image.shape[0]:
            print(f"Warning: Landmark outside the image at ({x_pred}, {y_pred})")

cv2.imshow("Predicted Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


