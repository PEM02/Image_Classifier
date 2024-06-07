# Image Classifier

This project is an image classifier designed to preprocess, segment, and classify images based on their features. The project uses MATLAB for image processing and clustering.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Contributing](#contributing)

## Introduction

The Image Classifier project processes a set of images by converting them to binary, segmenting objects, extracting features, and classifying them using various methods. This project is ideal for those interested in image processing and machine learning.

## Features

- Convert images to binary.
- Segment objects from binary images.
- Extract features such as area, perimeter, Euler number, and circularity.
- Cluster segmented objects using K-means clustering.
- Classify new images based on precomputed descriptors.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ImageClassifier.git
   cd ImageClassifier

2. **Ensure you have MATLAB installed:**
  Download and install MATLAB from MathWorks.(https://www.mathworks.com/)

## Usage

1. **Preprocess and segment images:**

  Place your images in the Bases Sossa folder.
  Run the MATLAB script to preprocess and segment the images:

    clc;
    clear;
    close all;
    % Ensure your script code is here
2. **Classify a new image:**

  Select an image through the dialog window that opens upon running the script.
  The script will classify the image based on precomputed descriptors and display the results.

## Functions

  **convertir_imagenes_a_binarias_en_carpeta(ruta_carpeta, umbral)**: Converts images to binary based on a threshold.
  
  **convertir_a_binaria(imagen, umbral)**: Converts a single image to binary.

  **leer_imagenes_en_carpeta(ruta_carpeta)**: Reads images from a specified folder.
  
  **extractObjects(binaryImage)**: Extracts objects from a binary image.
  
  **cropImage(binaryImage)**: Crops a binary image to the bounding box of the object.

  **processSegmentedImages(ruta_nueva_carpeta)**: Processes segmented images to extract descriptors and cluster them.
  
  **calculateArea(binaryImage)**: Calculates the area of a binary image.

  **calculatePerimeter(binaryImage)**: Calculates the perimeter of a binary image.

  **calculateEulerNumber(binaryImage)**: Calculates the Euler number of a binary image.

  **calculateCircularity(area, perimeter)**: Calculates the circularity of an object.

  **simpleClustering(data, k)**: Performs K-means clustering.

  **classifyImage(imagePath, descriptorsPath)**: Classifies an image using precomputed descriptors.

  **EuclideanToCentroids(descriptor, centroids, distanceThreshold)**: Classification function using Euclidean distance.
  
  **MahalanobisToCentroids(descriptor, data, idx, distanceThreshold)**: Classification function using Mahalanobis distance.

  **classifyByMaximumProbability(newImagedescritor, descriptor, idx, treshold)**: Classification function using maximum probability.

  **knn(descriptor, descriptors, idx, k, distanceThreshold)**: Classification function using K-nearest neighbors.

## Contributing
 
  Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

  -Fork the repository.
  -Create a new branch (git checkout -b feature-branch).
  -Commit your changes (git commit -am 'Add new feature').
  -Push to the branch (git push origin feature-branch).
  -Create a new pull request.



