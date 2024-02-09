# Visual Machine Learning for Intelligent Vehicle (UROP)

This project, developed as part of a summer research programme, is aimed at building a hands-on understanding and practical experience of the technology behind autonomous vehicles through the implementation of machine learning and computer vision systems. It encapsulates the development and integration of automated control and vision systems for self-driving vehicles, leveraging a combination of Python, PyTorch, OpenCV, Flask, and machine learning techniques.

## Project Components

### 1. Driving Behaviour Cloning

This component focuses on the simulation of autonomous driving behavior within a Unity-developed driving simulation environment. Utilizing a Convolutional Neural Network (CNN), the system predicts steering angles based on real-time visual input, facilitating automated vehicle control. Key achievements include:

- **API Implementation**: Developed a Flask-based API to integrate the trained CNN model, enabling real-time automated control over vehicle speed and steering within the Self-Driving Car Simulator environment.
  - [Driving Behaviour Cloning Folder](/CNN-Driving-Behaviour-Cloning)

### 2. Semantic Road Segmentation

The objective here is to accurately segment road from non-road regions within the KITTI road dataset, a critical task for autonomous navigation. A Fully Convolutional Network (FCN) architecture is employed to classify pixels into their respective semantic categories. Innovations and improvements in this area encompass:

- **Automated Hyper-parameter Tuning**: Implemented an automated process for fine-tuning the hyper-parameters of the FCN model, significantly enhancing the accuracy of road semantic segmentation.
- **Post-Processing and Performance Analysis**: Conducted thorough analyses of segmentation results, optimizing post-processing techniques to refine the segmentation output further.
  - [Semantic Road Segmentation Folder](/FCN-Road-Semantic-Segmentation)

### 3. Lane Detection

This section is dedicated to the detection of lane markings, a vital component for the safe operation of autonomous vehicles. Through the application of moviepy and OpenCV, this part of the project focuses on:

- **Video Processing**: Utilized moviepy alongside OpenCV for robust video processing, enabling the extraction and analysis of lane information from video data.
- **Modularization and Streamlining**: Modularized core image processing algorithms to improve code reusability and maintainability. Streamlined the lane detection workflow to enhance the efficiency and accuracy of lane detection in real-time.
  - [Lane Detection Folder](/OpenCV-Lane-Detection)

## Project Overview

For detailed information on each project component, please refer to the README.md files in the respective subfolders:

- [**Driving Behaviour Cloning**](/CNN-Driving-Behaviour-Cloning/README.md)
- [**Semantic Road Segmentation**](/FCN-Road-Semantic-Segmentation/README.md)
- [**Lane Detection**](/OpenCV-Lane-Detection/README.md)
