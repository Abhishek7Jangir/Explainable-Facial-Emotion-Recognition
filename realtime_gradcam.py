# realtime_explainable_emotion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import copy
import sys
import os

# -------- CONFIG --------
IMG_H, IMG_W = 48, 48
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = 'bestModelOnCleanDataset_1.pth'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# -------- MODEL (same as training) --------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # Block 2: (Input: 24x24x64) -> (Output: 12x12x128)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        
        # Block 3: (Input: 12x12x128) -> (Output: 6x6x256)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # Block 4: (Input: 6x6x256) -> (Output: 3x3x512)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        
        # Classifier Head: (Input: 3*3*512 = 4608)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Connect the blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        return x
# -------- setup device & models --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.isfile(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}", file=sys.stderr)
    sys.exit(1)

model = EmotionCNN(NUM_CLASSES).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()   # inference model (keeps BatchNorm in eval mode)

# cloned model used only for Grad-CAM backward pass (keeps main model untouched)
gradcam_model = copy.deepcopy(model).to(device)
gradcam_model.eval()

# -------- hooks for Grad-CAM --------
features = None
gradients = None
def forward_hook(module, inp, out):
    global features
    features = out
def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# target layer: the second ReLU inside conv_block4 is at index 5
target_layer = gradcam_model.conv_block4[5]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# -------- transforms (match validation in training script) --------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),   # no normalization because training used none
])

# -------- face detector --------
if not os.path.isfile(CASCADE_PATH):
    print(f"Face cascade not found: {CASCADE_PATH}", file=sys.stderr)
    sys.exit(1)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# -------- camera loop --------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam", file=sys.stderr)
    sys.exit(1)

print("Running. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

        for (x, y, w, h) in faces:
            # square crop (centered) to avoid distortion
            size = max(w, h)
            cx, cy = x + w // 2, y + h // 2
            x1, y1 = max(0, cx - size//2), max(0, cy - size//2)
            x2, y2 = x1 + size, y1 + size
            # clamp to frame
            x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
            x1 = max(0, x1); y1 = max(0, y1)
            roi_gray = gray[y1:y2, x1:x2]

            # if roi too small skip
            if roi_gray.size == 0:
                continue

            # preprocess exactly like validation
            roi_pil = Image.fromarray(roi_gray)
            img_t = transform(roi_pil).unsqueeze(0).to(device)   # shape (1,1,48,48)

            # ----- inference (pure, no grad) -----
            with torch.no_grad():
                out = model(img_t)                 # logits
                probs = F.softmax(out, dim=1)
                conf, pred = torch.max(probs, 1)
                label = CLASS_NAMES[pred.item()]
                conf_val = conf.item() * 100.0

            # ----- Grad-CAM on cloned model (isolated backward) -----
            # zero grads and run forward on gradcam_model
            gradcam_model.zero_grad()
            _out = gradcam_model(img_t)            # forward (this sets features via hook)
            score = _out[0, pred.item()]           # pick same predicted class
            score.backward()                       # compute gradients on cloned model

            # compute heatmap safely (detach before moving to numpy)
            if features is None or gradients is None:
                heatmap = None
            else:
                pooled_grads = torch.mean(gradients, dim=[0,2,3])   # shape (C,)
                f_map = features.clone().detach()                  # (1,C,H,W)
                for i in range(f_map.shape[1]):
                    f_map[0, i, :, :] *= pooled_grads[i].detach()
                heatmap = torch.mean(f_map[0], dim=0).detach().cpu().numpy()
                heatmap = np.maximum(heatmap, 0)
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()
                else:
                    heatmap = None

            # ----- overlay heatmap if exists -----
            if heatmap is not None:
                heatmap_resized = cv2.resize(heatmap, (x2-x1, y2-y1))
                heatmap_uint8 = np.uint8(255 * heatmap_resized)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                face_color = frame[y1:y2, x1:x2]
                # ensure same shape
                if face_color.shape[:2] == heatmap_color.shape[:2]:
                    blended = cv2.addWeighted(face_color, 0.6, heatmap_color, 0.4, 0)
                    frame[y1:y2, x1:x2] = blended

            # ----- draw label and box -----
            display_text = f"{label.capitalize()} ({conf_val:.1f}%)"
            cv2.putText(frame, display_text, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.imshow("Explainable Emotion Detector (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")
