import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset class
class SignLanguageDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data.iloc[index].values.astype(np.uint8).reshape((28, 28, 1))  # Reshape the image
        x = np.repeat(x, 3, axis=2)  # Convert to 3 channels
        y = self.target.iloc[index]
        if self.transform:
            x = self.transform(x)
        return x, y

# Read the data
train_df = pd.read_csv('sign_mnist_train.csv')

# Split the labels and the image data
y = train_df['label']
X = train_df.drop(['label'], axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Define the transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),  # Resize the images to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create the data loaders
train_dataset = SignLanguageDataset(X_train, y_train, transform)
val_dataset = SignLanguageDataset(X_val, y_val, transform)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the model
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, len(y.unique()))

# Adding dropout layer
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, len(y.unique()))
)

# Make sure to move the model to device after modifications
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# For storing metrics for each epoch
train_losses, val_losses, train_accs, val_accs = [], [], [], []

# Use a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# Add early stopping
min_val_loss = np.Inf
n_epochs_stop = 5
epochs_no_improve = 0

# Train the model
model = model.to(device)
for epoch in range(15):
    running_train_loss = 0.0
    running_val_loss = 0.0
    running_train_corrects = 0
    running_val_corrects = 0
    
    model.train()  # Set model to training mode
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)
        running_train_corrects += torch.sum(preds == labels.data)

    model.eval()  # Set model to evaluate mode
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * inputs.size(0)
            running_val_corrects += torch.sum(preds == labels.data)

    epoch_train_loss = running_train_loss / len(trainloader.dataset)
    epoch_val_loss = running_val_loss / len(valloader.dataset)
    epoch_train_acc = running_train_corrects.double() / len(trainloader.dataset)
    epoch_val_acc = running_val_corrects.double() / len(valloader.dataset)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accs.append(epoch_train_acc)
    val_accs.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{15}, Train Loss: {epoch_train_loss}, Train Acc: {epoch_train_acc}, Val Loss: {epoch_val_loss}, Val Acc: {epoch_val_acc}")

print('Finished Training')

# Plotting loss
plt.figure(figsize=(10, 7))
plt.plot(train_losses, color='orange', label='train loss')
plt.plot(val_losses, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plotting accuracy
plt.figure(figsize=(10, 7))
plt.plot(train_accs, color='green', label='train accuracy')
plt.plot(val_accs, color='blue', label='validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), 'model.pth')