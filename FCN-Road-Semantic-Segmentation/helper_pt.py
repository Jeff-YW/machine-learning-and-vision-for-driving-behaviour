import shutil
import torch
import re
import time
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob
import numpy as np
import torchvision
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image


def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) for two binary masks.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def results_eval(result_folder, gt_folder):
    ious = []

    pred_masks = sorted(glob(os.path.join(result_folder, '*.png')))
    gt_masks = sorted(glob(os.path.join(gt_folder, 'gt_image_2', '*.png')))

    for pred_mask_path, gt_mask_path in zip(pred_masks, gt_masks):
        # Ensure we are comparing masks with the same name
        assert os.path.basename(pred_mask_path) == os.path.basename(gt_mask_path), \
            "The mask names are not matching! Make sure they are in order."

        pred_mask = np.array(Image.open(pred_mask_path))
        gt_mask = np.array(Image.open(gt_mask_path))

        # Convert masks to binary format. This step might need adjustment based on your dataset.
        pred_mask_binary = (pred_mask > 127).astype(np.uint8)
        gt_mask_binary = (gt_mask > 127).astype(np.uint8)

        iou = calculate_iou(pred_mask_binary, gt_mask_binary)
        ious.append(iou)

    # Calculate average IoU
    average_iou = np.mean(ious)
    return average_iou

def gen_test_output(model, data_folder, image_shape, device):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor()
    ])

    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = Image.open(image_file)
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference
        logits = model(image_tensor)
        im_softmax = torch.nn.functional.softmax(logits, dim=1)
        # Splice out second column (road), reshape output back to image_shape
        im_softmax = im_softmax[0][1]
        # If road softmax > 0.5, prediction is road
        segmentation = (im_softmax > 0.5).cpu().numpy()
        # Create mask based on segmentation
        mask = np.dot(segmentation[..., None], np.array([[0, 255, 0, 127]]))
        mask = Image.fromarray(mask, 'RGBA')
        street_im = Image.open(image_file).convert('RGBA')
        street_im.paste(mask, box=None, mask=mask)

        # returns a generator object
        yield os.path.basename(image_file), street_im   # essentially the name of the file without the directory path


def save_inference_samples(runs_dir, data_dir, model, image_shape, device):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))

    # Delete an entire directory tree
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to:', output_dir)
    model.eval()  # set model to evaluation mode
    image_outputs = gen_test_output(model, os.path.join(data_dir, 'data_road/testing'), image_shape, device)

    # iterate the tuple (generator objects ...)
    for name, image in image_outputs:
        image.save(os.path.join(output_dir, name))

    return output_dir


class KittiDataset(Dataset):
    def __init__(self, data_folder, image_shape, transform=None):
        self.image_shape = image_shape
        self.transform = transform

        # Grab image and label paths
        self.image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        self.label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))
        }

        self.background_color = np.array([255, 0, 0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_file = self.image_paths[idx]
        gt_image_file = self.label_paths[os.path.basename(image_file)]

        # Load images using PIL
        image = Image.open(image_file).resize(self.image_shape)
        gt_image = Image.open(gt_image_file).resize(self.image_shape)

        # Randomly flip the image along y-axis
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            gt_image = gt_image.transpose(Image.FLIP_LEFT_RIGHT)

        image = np.array(image)
        gt_image = np.array(gt_image)

        # Create "one-hot-like" labels by class
        gt_bg = np.all(gt_image == self.background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

        if self.transform:
            save_inference_samples            image = self.transform(image)
            gt_image = self.transform(gt_image)
            image = image.float()
            gt_image = gt_image.float()

        return image, gt_image
