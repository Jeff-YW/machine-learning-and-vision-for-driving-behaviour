import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NvidiaModel(nn.Module):
    def __init__(self, img_h, img_w, img_d, dropout=0.5):
        # input shape originally is 160 * 320 * 3
        super(NvidiaModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=img_d, out_channels=24, kernel_size=(5, 5), stride=(2, 2))
        self.conv2 = nn.Conv2d(24, 36, (5, 5), stride=(2, 2))
        self.conv3 = nn.Conv2d(36, 48, (5, 5), stride=(2, 2))
        self.conv4 = nn.Conv2d(48, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 64, (3, 3))
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Flattening layer is implicit in forward method
        # Fully connected layers
        # need to calculate the correct in_features for fc1 based on the conv layers' output size based on the shape!
        self.fc1 = nn.Linear(in_features=27456, out_features=200)  # Modify in_features
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        # Activation Layer
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.elu(self.conv5(x))
        x = x.view(x.size(0), -1)  # Flatten the output for the dense layer
        x = self.dropout(self.elu(self.fc1(x)))
        x = self.dropout(self.elu(self.fc2(x)))
        x = self.elu(self.fc3(x))
        x = self.fc4(x)
        return x