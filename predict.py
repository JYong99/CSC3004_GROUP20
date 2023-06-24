import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import string
import matplotlib.pyplot as plt
import seaborn as sn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SignLanguageDataset(Dataset):
    def init(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def len(self):
        return len(self.data)

    def getitem(self, index):
        x = self.data.iloc[index].values.astype(np.uint8).reshape((28, 28))  # Reshape the image
        x = Image.fromarray(x)  # Convert to PIL Image
        y = self.target.iloc[index]
        if self.transform:
            x = self.transform(x)
        return x, y

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Define the label map
label_map = {i: letter for i, letter in enumerate(string.ascii_uppercase)}
label_map[24] = "del"
label_map[25] = "nothing"
label_map[26] = "space"

# Load pre-trained ResNet50 and reset final fully connected layer
model = models.resnet50(pretrained=False) 
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(model.fc.in_features, 25)
model.load_state_dict(torch.load('sign_language_resnet50.pth', map_location=device)) 
model = model.to(device)

# Initialize the prediction and label lists(tensors)
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

def predict_letter_single_image(image, model):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        print('Predicted label:', label_map[predicted.item()])
    return label_map[predicted.item()]

# Assume 'image_path' is the path to the image file you wish to classify
image_path = 'test_image1_C.png'

# Open your image with PIL's Image module and convert it to grayscale
image = Image.open(image_path).convert("L")

# Apply the transformation to your image, this makes sure it is the correct shape and size
image_tensor = val_transform(image)

# Add an additional dimension to the start of the tensor, this is because
# PyTorch's models expect a batch of images as input, even if it's only one image
image_tensor = image_tensor.unsqueeze(0)

predict_letter_single_image(image_tensor, model)