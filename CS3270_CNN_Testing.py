# load_and_predict.py

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extract= nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,16,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,16,kernel_size=3, padding =1),
            nn.ReLU(),
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*55*55, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,x):
        x = self.feature_extract(x)
        x = self.bn1(x)
        x = self.classifier(x)
        return x


model = CNN_Model().to(device)
model.load_state_dict(torch.load("cat_cnn.pth", map_location=device))
model.eval()

print("Model loaded successfully!")


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


try:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to stream")
            break

        # Convert OpenCV frame (BGR) to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to match training input
        gray = cv2.resize(gray, (224, 224))

        # Convert to PyTorch tensor and normalize to [0, 1]
        img = torch.from_numpy(gray).float().div(255.0)

        # Add channel and batch dimensions: [1, 1, 224, 224]
        img = img.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            probabilities = torch.exp(output)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        confidence_percent = confidence.item() * 100

        # Display prediction on camera frame
        cv2.putText(
            frame,
            f"Class: {predicted_class} | Confidence: {confidence_percent:.2f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("CNN Camera Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()


#predict_folder("cats/TestCats")