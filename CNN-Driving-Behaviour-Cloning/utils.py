from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import matplotlib.image as mpimg
import torch

# Initialize the model
img_h, img_w, img_d = 160, 320, 3  # Example dimensions, adjust accordingly

class DrivingDataset(Dataset):
    def __init__(self, dataframe, transform=None, training=True, img_folder_path=None):
        self.dataframe = dataframe
        self.transform = transform
        self.training = training  # flag for train/validation dataset
        self.img_folder_path = img_folder_path

    def __len__(self):
        return len(self.dataframe)

    def preprocess(self, image):
        img_crop = image[60:-25, :, :]
        img_resize = cv2.resize(img_crop, (img_w, img_h), cv2.INTER_AREA)
        img_yuv = cv2.cvtColor(img_resize, cv2.COLOR_RGB2YUV)
        return img_yuv

    def read_data(self, sample):
        """
        Read data in the normal way:
        read the image from central camera and the steering angle without revision
        """
        org_img_path = sample[0] # sample.center
        img_name = org_img_path.split('/')[-1]
        img = mpimg.imread(self.img_folder_path + "/IMG/" + img_name)
        ang = sample[3]  # sample.steering
        return img, ang

    def rand_read_data(self, sample):
        """
        Read data in random ways:
        1. Left: 1/6
        2. Flipped left: 1/6
        3. Right: 1/6
        4. Flipped right: 1/6
        5. Center: 1/6
        6. Flipped center: 1/6
        """
        ran_num = np.random.rand()
        if ran_num < 0.33:
            org_img_path = sample[1]  # sample.left
            img_name = org_img_path.split('/')[-1]
            img = mpimg.imread(self.img_folder_path + "/IMG/" + img_name)
            ang = sample[3] + 0.2

        elif ran_num < 0.66:
            org_img_path = sample[2]   # sample.right
            img_name = org_img_path.split('/')[-1]
            img = mpimg.imread(self.img_folder_path + "/IMG/" + img_name)
            ang = sample[3] - 0.2

        else:
            org_img_path = sample[0]  # sample.center
            img_name = org_img_path.split('/')[-1]
            img = mpimg.imread(self.img_folder_path + "/IMG/" + img_name)
            ang = sample[3]

        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            ang = -ang

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * (0.4 * np.random.rand() + 0.8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img, ang

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        # randomly flip / shift the image to augment the input dataset
        if self.training and np.random.rand() < 0.75:
            img, ang = self.rand_read_data(sample)
        else:
            img, ang = self.read_data(sample)
        img = self.preprocess(img)

        # Convert img and ang to tensors
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)  # Rearrange channels for PyTorch (C, H, W)
        ang = torch.tensor(ang, dtype=torch.float)
        return img, ang


def preprocess(image):
    """
    Crop, resize and convert the colorspace of the image.
    """
    # Crop the image to remove the sky and the hood of the car
    img_crop = image[60:-25, :, :]
    # Resize the image to 160x320x3 to recover the resolution and make the features more evident
    img_resize = cv2.resize(img_crop, (img_w, img_h), cv2.INTER_AREA)
    # Convert the image to YUV color space (Ref: Nvidia Paper)
    img_yuv = cv2.cvtColor(img_resize, cv2.COLOR_RGB2YUV)
    return torch.tensor(img_yuv)