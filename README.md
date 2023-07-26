# AISC2013_Ex3
DeepXRay: Deep Learning for Lung X-ray Classification of Non-Covid and Covid infected

# DeepXRay: Deep Learning for Lung X-ray Classification of Non-Covid and Covid Infected

![Lung X-ray](https://github.com/Meenakshi-Remadevi/Deployment-of-AI/blob/main/eXtmE1V2XgsjZK2JolVQ5g_Border_of_left_atrium.jpeg)

## Table of Contents
1. [Introduction](#introduction)
2. [Image Dataset Preparation for Lung X-ray Classification](#image-dataset-preparation-for-lung-x-ray-classification)
3. [CNN Model for Lung X-ray Classification](#cnn-model-for-lung-x-ray-classification)
4. [DNN Model for Lung X-ray Classification](#dnn-model-for-lung-x-ray-classification)
5. [Contributors](#contributors)
6. [Next Steps](#next-steps)

## Introduction
The DeepXRay project aims to use deep learning techniques to classify lung X-ray images as either Non-Covid or Covid infected. This project was developed as part of the AISC2013 - Deployment of AI Solutions 02, In-class Application Exercise 3, on 26 July 2023.

## Image Dataset Preparation for Lung X-ray Classification
This section outlines the steps taken to prepare the image dataset for lung X-ray classification. It includes data preparation, augmentation, splitting, and visualization of the dataset.

### Data Preparation & Augmentation
The image dataset is organized, and class labels are assigned (0 for normal, 1 for COVID-19, and 2 for non-COVID). Images are processed by resizing, rotating, and blurring to augment the dataset and improve model generalization.

### Data Splitting & Visualization
The dataset is split into training, validation, and test sets. The augmented data is visualized to understand class distributions.

## CNN Model for Lung X-ray Classification
This section focuses on the Convolutional Neural Network (CNN) model created for lung X-ray classification.

### Data Preparation & Model Creation
Target data is converted to one-hot encoded format. The CNN model consists of convolutional, max pooling, batch normalization, and dropout layers, with an output layer having 3 units for the 3 classes. The model is compiled using the Adam optimizer and categorical cross-entropy loss.

### Training the CNN Model
The CNN model is trained on the training data using a batch size of 64 and 10 epochs, with 10% validation split for monitoring.

### CNN Model Evaluation & Performance Metrics
The CNN model is evaluated on the test data for accuracy. Confusion matrix, classification report, and AUC score are generated to assess the model's performance.

## DNN Model for Lung X-ray Classification
This section focuses on the Deep Neural Network (DNN) model created for lung X-ray classification.

### Data Preparation & Model Creation
Similar to the CNN model, target data is converted to one-hot encoded format. The DNN model includes dense layers, batch normalization, and dropout, with an output layer having 3 units for the 3 classes. The model is compiled using the Adam optimizer and categorical cross-entropy loss.

### Training the DNN Model
The DNN model is trained on the training data using a batch size of 64 and 10 epochs, with 10% validation split for monitoring.

### DNN Model Evaluation & Performance Metrics
The DNN model is evaluated on the test data for accuracy. Confusion matrix, classification report, and AUC score are generated to assess the model's performance.

## Contributors
- Akshay Rajesh Krishna (500209511)
- Akshay Ajeesh (500209543)
- Meenakshi Remadevi (500209913)

## Next Steps
Both CNN and DNN models demonstrate promising performance for lung X-ray classification. To further optimize the models and enhance the project, the following steps are suggested:
- Fine-tune hyperparameters and architecture.
- Explore other deep learning techniques and data augmentation strategies.
- Evaluate models on larger datasets for robustness.
- Deploy the best model for practical applications.

Thank you for your interest in the DeepXRay project!

Group C

---
