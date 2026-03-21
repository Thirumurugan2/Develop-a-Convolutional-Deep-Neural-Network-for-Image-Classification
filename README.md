# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images
## Neural Network Model
<img width="909" height="639" alt="Screenshot 2026-02-27 101952" src="https://github.com/user-attachments/assets/cdfd0070-23f6-4f3e-b336-5f326376289c" />


## DESIGN STEPS
### STEP 1: 
Load the Fashion-MNIST dataset and preprocess images using tensor conversion and normalization.

### STEP 2: 
Create DataLoaders for training and testing with a fixed batch size.


### STEP 3: 
Design a CNN model with convolution, ReLU activation, max-pooling, and fully connected layers.



### STEP 4: 
Train the model using Cross-Entropy loss and the Adam optimizer for several epochs.


### STEP 5: 
Test the trained model to calculate accuracy, confusion matrix, and classification report.


### STEP 6: 
Predict and display the class of a single test image with actual and predicted labels.




## PROGRAM
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

# Get the shape of the first image in the training dataset
image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))

# Get the shape of the first image in the test dataset
image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x

from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name: THIRUMURUGAN R')
print('Register Number: 212223220118')
summary(model, input_size=(1, 28, 28))

# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=3):

    # write your code here
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Name:  MOHAN KRISHNA J')
        print('Register Number: 212223220060')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Train the model
train_model(model, train_loader)

## Step 4: Test the Model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: THIRUMURUGAN R')
    print('Register Number: 212223220118')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name: THIRUMURUGAN R')
    print('Register Number: 212223220118')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name: THIRUMURUGAN R')
    print('Register Number: 212223220118')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Evaluate the model
test_model(model, test_loader)

## Step 5: Predict on a Single Image
import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes

    # Display the image
    print('Name: THIRUMURUGAN')
    print('Register Number: 212223220118')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')

# Example Prediction
predict_image(model, image_index=190, dataset=test_dataset)

```

### OUTPUT

## Training Loss per Epoch
![image alt](https://github.com/MohanKrishnaJ/Develop-a-Convolutional-Deep-Neural-Network-for-Image-Classification/blob/8acf18143139aded393ecf108a3105fe618dedf9/Screenshot%202026-03-16%20011919.png)

## Confusion Matrix
![image alt](https://github.com/MohanKrishnaJ/Develop-a-Convolutional-Deep-Neural-Network-for-Image-Classification/blob/0c431c459093690341df3f038307fa2267ee23bb/Screenshot%202026-03-16%20012247.png)

## Classification Report

![image alt](https://github.com/MohanKrishnaJ/Develop-a-Convolutional-Deep-Neural-Network-for-Image-Classification/blob/a0fd549670aa27b1e4e71ed0995f5e1bf8f2e00e/Screenshot%202026-03-16%20012407.png)
### New Sample Data Prediction

![image alt](https://github.com/MohanKrishnaJ/Develop-a-Convolutional-Deep-Neural-Network-for-Image-Classification/blob/0481ff25b076eb192c8dc3b540612137b0d43ad2/Screenshot%202026-03-16%20012519.png)
## RESULT
This program has been executed successfully.
