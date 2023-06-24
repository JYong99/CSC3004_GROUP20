import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import string
from sklearn.metrics import confusion_matrix


device = torch.device("cuda")

class SignLanguageDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data.iloc[index].values.astype(np.uint8).reshape((28, 28))  # Reshape the image
        x = Image.fromarray(x)  # Convert to PIL Image
        y = self.target.iloc[index]
        if self.transform:
            x = self.transform(x)
        return x, y

val_transform = transforms.Compose([
    transforms.Resize(224),
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
model.fc = nn.Linear(model.fc.in_features, 27)
model.load_state_dict(torch.load('sign_language_resnet50.pth'))
model = model.to(device)

# Initialize the prediction and label lists(tensors)
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

def predict_letter(data_loader, model, num_classes):
    model.eval()
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(data_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Append batch prediction results
            predlist=torch.cat([predlist,predicted.view(-1).cpu()])
            lbllist=torch.cat([lbllist,classes.view(-1).cpu()])

    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    # Expand the confusion matrix to the full number of classes
    conf_mat_full = np.zeros((num_classes, num_classes), dtype=int)
    unique_labels = np.unique(lbllist.numpy())
    conf_mat_full[np.ix_(unique_labels, unique_labels)] = conf_mat

    # Print the confusion matrix
    print(conf_mat_full)

# we need to make a DataLoader for the test set
test_df = pd.read_csv('sign_mnist_test.csv')
test_dataset = SignLanguageDataset(test_df.iloc[:, 1:], test_df.iloc[:, 0], transform=val_transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

predict_letter(test_dataloader, model, 27)  # Assuming your model was trained on 27 classes
