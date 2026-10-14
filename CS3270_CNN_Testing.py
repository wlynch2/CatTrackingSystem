# load_and_predict.py

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt


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
            nn.Linear(256,3),
            nn.Dropout(0.5),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,x):
        x = self.feature_extract(x)
        x = self.bn1(x)
        x = self.classifier(x)
        return x


model = CNN_Model().to(device)
model.load_state_dict(torch.load("mnist_cnn_model.pth", map_location=device))
model.eval()

print("Model loaded successfully!")


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png','.jpg','.jpeg')):
            img = Image.open(os.path.join(folder_path, filename))
            img = transform(img).unsqueeze(0).to(device)

            output = model(img)
            _, predicted = torch.max(output, 1)

            print(f"{filename} -> Predicted Digit: {predicted.item()}")

            img_display = img.squeeze().cpu().numpy()

            plt.figure()
            plt.imshow(img_display, cmap='gray')
            plt.title(f"Predicted class: {predicted.item()}")
            plt.axis("off")
            plt.show()

predict_folder("cats/TestCats")