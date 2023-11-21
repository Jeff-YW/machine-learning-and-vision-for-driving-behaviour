'''
This file is used for unit testing your work within main.py.
'''

import sys
import os
# from copy import deepcopy
from glob import glob
# from unittest import mock
# import numpy as np
import torch


def _assert_tensor_shape(tensor, target_shape, name):
    """
    Check if tensor's shape matches target_shape
    """
    assert tuple(tensor.shape) == tuple(target_shape), \
        f'{name} tensor has shape {tensor.shape}, but expected {target_shape}.'

def test_layers(fcn_model):
    """
    Test whether the PyTorch FCN model generates output with the correct shape.
    :param fcn_model: An instance of FCNVGG16.
    """
    num_classes = fcn_model.conv1x1_3.out_channels  # extract the number of classes from the model

    # Create a dummy input image
    input_image = torch.randn(1, 3, 160, 576)  # [batch_size, channels, height, width]

    # Get the output of the FCN model
    layers_output = fcn_model(input_image)

    _assert_tensor_shape(layers_output, [1, num_classes, 160, 576], 'Layers Output')


def test_for_kitti_dataset(data_dir):
    """
    Test whether the KITTI dataset has been downloaded, and whether the full, correct dataset is present.
    :param data_dir: Directory where the KITTI dataset was downloaded into.
    """
    kitti_dataset_path = os.path.join(data_dir, 'data_road')
    training_labels_count = len(glob(os.path.join(kitti_dataset_path, 'training/gt_image_2/*_road_*.png')))
    training_images_count = len(glob(os.path.join(kitti_dataset_path, 'training/image_2/*.png')))
    testing_images_count = len(glob(os.path.join(kitti_dataset_path, 'testing/image_2/*.png')))

    assert not (training_images_count == training_labels_count == testing_images_count == 0),\
        'Kitti dataset not found. Extract Kitti dataset in {}'.format(kitti_dataset_path)
    assert training_images_count == 289, 'Expected 289 training images, found {} images.'.format(training_images_count)
    assert training_labels_count == 289, 'Expected 289 training labels, found {} labels.'.format(training_labels_count)
    assert testing_images_count == 290, 'Expected 290 testing images, found {} images.'.format(testing_images_count)