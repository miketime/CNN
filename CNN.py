import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
import os




def load_images_and_labels(image_folder, label_folder, image_ext='.jpg', label_ext='.txt'):
    image_paths = []
    labels = []
    # List all files in the image folder
    for filename in os.listdir(image_folder):
        if filename.endswith(image_ext):
            # Construct the full path to the image file
            img_path = os.path.join(image_folder, filename)
            # Construct the corresponding label file path
            label_filename = filename.replace(image_ext, label_ext)
            label_path = os.path.join(label_folder, label_filename)
            # Append paths
            image_paths.append(img_path)
            labels.append(label_path)
    return image_paths, labels

# Example usage
image_folder = 'D:\Facultate\Licenta\CNN-Github\img_labels\train\images'
label_folder = 'D:\Facultate\Licenta\CNN-Github\img_labels\train\labels'  # This can be the same as image_folder if labels are in the same folder
image_paths, label_paths = load_images_and_labels(image_folder, label_folder)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (416, 416))
        if self.transform:
            image = self.transform(image)

        with open(label_folder, 'r') as file:
            label = file.read().strip()

        return image, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = CustomDataset(image_paths, label_paths, transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
