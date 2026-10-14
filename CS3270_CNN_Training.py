
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

print("begin")

def create_dataset_loader(path ,transform=None, batch_size=64, shuffle=True):
    dataset = datasets.ImageFolder(path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomAutocontrast(),
    transforms.RandomRotation(25),
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

path = "cats"
train_path = path+"/training_set"
test_path = path+"/test_set"

dataloader_train = create_dataset_loader(train_path, train_transform)
dataloader_test = create_dataset_loader(test_path, test_transform)


class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extract= nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(16,16,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
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
            nn.Linear(256,3),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,x):
        x = self.feature_extract(x)
        x = self.bn1(x)
        x = self.classifier(x)
        return x

model = CNN_Model().to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

num_epochs = 30

for epoch in range(num_epochs):

    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for img, label in dataloader_train:
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    train_loss = running_loss / len(dataloader_train)
    train_acc = correct / total

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in dataloader_test:
            img, label = img.to(device), label.to(device)

            output = model(img)
            loss = criterion(output, label)

            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    test_loss = test_loss / len(dataloader_test)
    test_acc = correct / total

    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    scheduler.step(test_loss)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")
    print("-"*40)

import matplotlib.pyplot as plt

epochs = range(1, num_epochs + 1)


plt.figure()
plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, test_losses, label="Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Testing Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(epochs, train_accuracies, label="Training Accuracy")
plt.plot(epochs, test_accuracies, label="Testing Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.legend()
plt.grid()
plt.show()


torch.save(model.state_dict(), "cat_cnn.pth")
print("Model saved successfully!")