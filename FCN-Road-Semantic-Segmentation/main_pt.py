import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2

import helper_pt as helper
import project_tests_pt as tests
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class FCNVGG16(nn.Module):
    def __init__(self, num_classes):
        super(FCNVGG16, self).__init__()

        # Load pre-trained VGG16
        # vgg = models.vgg16(pretrained=True)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        # Extract features from VGG's layers 3, 4, and 7
        self.features3 = nn.Sequential(*features[:16])
        self.features4 = nn.Sequential(*features[16:23])
        self.features7 = nn.Sequential(*features[23:30])

        # 1x1 convolutions
        self.conv1x1_3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.conv1x1_7 = nn.Conv2d(512, num_classes, kernel_size=1)

        # Upsampling layers
        self.deconv1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        # self.deconv3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)
        self.deconv3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=4, padding=1, output_padding=2)

    def forward(self, x):

        # create a FCN using vgg 3rd, 4th and 7th layers

        x3 = self.features3(x)
        x4 = self.features4(x3)
        x7 = self.features7(x4)

        # Apply 1x1 convolutions
        x3 = self.conv1x1_3(x3)
        x4 = self.conv1x1_4(x4)
        x7 = self.conv1x1_7(x7)

        # Upsample and add skip connections
        x = self.deconv1(x7) + x4
        x = self.deconv2(x) + x3
        x = self.deconv3(x)

        return x


def run(epoch_num, batch_size, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Run on device {device}")

    # Initialization
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = 'data'
    runs_dir = 'runs'

    # Check if the runs directory exists, and create it if it doesn't
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    tests.test_for_kitti_dataset(data_dir)

    # Optional: Training on Cityscapes Dataset
    # ...

    # transforms.Compose expects a list of instantiated transform objects, not the transform classes themselves
    # Transformation =torchvision.transforms.Compose([torchvision.transforms.ToTensor])
    Transformation = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Create torch utils class to get batches of images
    dataset = helper.KittiDataset(os.path.join(data_dir, 'data_road/training'), image_shape=image_shape, transform=Transformation)

    # Assuming dataset is already defined
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load the FCN model based on VGG16 with pretrained weights
    fcn_model = FCNVGG16(num_classes=num_classes).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fcn_model.parameters(), lr=0.001)  # Learning rate placeholder

    # Lists for logging and visualization
    samples_plot = []
    loss_plot = []
    sample = 0

    print("Training...")

    # Train the model
    for epoch in range(epoch_num):
        # Training phase
        fcn_model.train()
        for inputs, labels in train_dataloader:
            # convert images to float type rather than unsigned char...
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = fcn_model(inputs)
            # custom loss function for batch input (N, C, W, H)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update logging lists
            sample += batch_size
            samples_plot.append(sample)
            loss_plot.append(loss.item())

            # Print progress
            print(f"Epoch {epoch + 1}/{epoch_num}, Batch {sample//batch_size}: Loss {loss.item():.4f}")

        # Validation phase
        fcn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = fcn_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{epoch_num}: Average Validation Loss {avg_val_loss:.4f}")

    # Save training information for visualization
    data_frame = pd.DataFrame(data={'sample': samples_plot, 'loss': loss_plot})
    data_frame.to_csv('train_information.csv')
    print('Train information saved.')

    # Check if the runs directory exists, and create it if it doesn't
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    # Save the trained model
    torch.save(fcn_model.state_dict(), os.path.join(runs_dir, "model.pth"))

    # Use the trained model to generate inference samples (assuming we have such a function)
    output_dir = helper.save_inference_samples(runs_dir, data_dir, fcn_model, image_shape, device)

    # evaluate the Jaccard Index (Intersection over Union, IoU)
    print("Average IoU is:", helper.results_eval(output_dir, data_dir))


if __name__ == '__main__':

    Epoch_Num = 1
    Batch_Size = 128
    # road or non-road
    Num_Classes = 2

    if os.path.exists("./run"):
        print("Yes")

    run(Epoch_Num, Batch_Size, Num_Classes)





