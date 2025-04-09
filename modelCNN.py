

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        num_classes = 2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class traintest():
    def train(model, train_data, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)

        for epoch in range(epochs):
            model.train()  # training mode activates

            for images, labels in train_data:  
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels) 
                loss.backward()
                optimizer.step()


    def test(model, test_data):
        model.eval()  # evaluation mode activates
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad(): 
            for images, labels in test_data:  ##
                images, labels = images.to("cpu"), labels.to("cpu")  
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)  

                _, predicted = torch.max(outputs, 1) 
                batch_accuracy = accuracy_score(labels.numpy(), predicted.numpy())
                total_accuracy += batch_accuracy * labels.size(0)  

                total_samples += labels.size(0)

        overall_loss = (total_loss / total_samples)
        overall_accuracy = (total_accuracy / total_samples) * 100

        print(f'Loss on test set: {overall_loss:.4f}, Accuracy on test set: {overall_accuracy:.2f}%')

        return overall_loss, overall_accuracy  