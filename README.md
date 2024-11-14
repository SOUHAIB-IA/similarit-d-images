# Image Recognition using K-NN and Histogram of Oriented Gradients (HOG)

This project implements an image recognition system using the K-Nearest Neighbors (K-NN) algorithm along with the Histogram of Oriented Gradients (HOG) descriptor for feature extraction. The system was developed to analyze, process, and recognize images based on feature similarity. The project includes image pre-processing, feature extraction, Euclidean distance calculations, and visualization of similar images.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Skills Utilized](#skills-utilized)


## Overview
The image recognition system uses HOG descriptors to convert image data into a vector of features, capturing essential gradients and edge patterns in the image. These features are then compared using Euclidean distance, enabling the K-NN algorithm to identify and retrieve the most similar images from a dataset.

## Features
- **Image Pre-processing**: Filtering and normalizing images to ensure consistency.
- **Feature Extraction**: Using HOG descriptors to represent the gradient structure of each image.
- **Image Similarity Calculation**: Using Euclidean distance to measure similarity between images.
- **Visualization**: Displaying the query image and the most similar images from the dataset.

## Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Install the necessary libraries:
   ```bash
   pip install numpy scipy matplotlib 
   ```

## Usage
1. Prepare your image dataset and organize it into training and testing folders.

## Dataset
- The dataset should be structured in folders for training and testing images.
- Pre-processing steps such as filtering and normalization ensure the images are uniform and optimized for feature extraction.

## Skills Utilized
- **Image Processing & Feature Extraction**: Using Histogram of Oriented Gradients (HOG) to generate feature descriptors.
- **Data Cleaning & Pre-processing**: Filtering and normalization techniques to prepare image data.
- **Data Visualization**: Displaying similar images using Python libraries (Matplotlib).
- **Python Libraries**: Hands-on experience with NumPy, SciPy, and scikit-image for image processing.
