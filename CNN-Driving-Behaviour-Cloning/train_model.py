from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from nvidia_model import NvidiaModel
from sklearn.model_selection import train_test_split
from utils import DrivingDataset
import os

# using term 1 simulator (oldest version)
# read csv file from left to right is:
# center img, left img, right_img, steering_angle, throttle, brake, speed


# Initialize the model
img_h, img_w, img_d = 160, 320, 3  # Example dimensions, adjust accordingly
batch_size = 80
epochs = 500
learning_rate = 0.0001

def save_checkpoint(epoch, model, optimizer, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)


if __name__ == "__main__":

    path_to_csv = "Training_dataset_2/driving_log.csv"      # Assuming '/Training_dataset_2/driving_log.csv' is accessible
    samples = pd.read_csv(path_to_csv)
    samples_train, samples_valid = train_test_split(samples, test_size=0.8, random_state=0)

    curr_wd_path = os.getcwd()
    folder_name = path_to_csv.split('/')[0] # Taking the most common folder name
    image_folder_path = curr_wd_path + "/" + folder_name

    train_dataset = DrivingDataset(samples_train, training=True, img_folder_path=image_folder_path)
    valid_dataset = DrivingDataset(samples_valid, training=False, img_folder_path=image_folder_path)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device}!")
    model = NvidiaModel(img_h, img_w, img_d).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):  # 500 epochs
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(dim=1), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation step
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(dim=1), targets)
                val_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Val Loss: {epoch_val_loss}")

        if (epoch + 1) % 50 == 0:
            # Checkpointing
            save_checkpoint(epoch, model, optimizer, f'model{epoch:03d}.pt')
