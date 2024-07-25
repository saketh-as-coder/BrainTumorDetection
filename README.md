# Brain Tumor Detection

## Introduction:
This project aims to develop a machine learning model that can detect and identify the type of tumor. The classifier uses a convolutional neural network (CNN) to analyze the MRI report submitted by the user.

## Data Preprocessing:
To ensure uniformity among the images, we resized all of them to a consistent dimension of 150 x 150 pixels. Image is represented as an array with the shape (1, 150, 150, 3),where 1 indicates that one image is there in array,150,150 represent the pixels and 3 corresponding to the RGB color channels.Each matrix within the array separately manages the red, green, and blue pixel components.

## Data Collection:
I took help of the Brain tumor classification dataset(https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) available on kaggle. 

## Why CNN is Used in This Project:
Convolutional Neural Networks (CNNs) are employed in this project due to their superior ability to handle image data. CNNs excel at recognizing patterns and features in images through their convolutional layers, which apply various filters to the input images to detect edges, textures, and other important details. This makes CNNs particularly well-suited for image classification and recognition tasks. Their architecture, which includes pooling layers to reduce dimensionality and fully connected layers for classification, enables efficient processing and accurate interpretation of visual data. The inherent ability of CNNs to learn spatial hierarchies of features further enhances their effectiveness in tasks involving image data.

## Result:
The accuracy and loss plots clearly indicate that the model is accurate enough upto 90% in analyzing the tumor from MRI reports.

 

