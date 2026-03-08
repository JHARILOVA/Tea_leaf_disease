

<img width="754" height="590" alt="matrix" src="https://github.com/user-attachments/assets/34a4d2c1-2b96-4ed6-a7e1-d7f4d3f882e7" />

# Tea Leaf Disease

The Tea Leaf Disease Dataset provides a collection of high-quality images aimed at identifying and classifying various diseases in tea leaves. Accurate detection of plant diseases is crucial in agriculture to ensure timely treatment and minimize crop loss. This dataset contains categorized images of tea leaves affected by different types of diseases such as Anthracnose, Algal Leaf, Bird Eye Spot, and others, as well as healthy leaves.

The goal of this notebook is to explore the dataset, perform image preprocessing, and build a robust machine learning model capable of classifying tea leaf diseases from images. The insights gained can assist in developing intelligent agricultural systems and mobile-based diagnosis tools for farmers.


# Project Overview

Tea is one of the most consumed beverages globally, but tea plants are highly susceptible to various leaf diseases that significantly impact yield and quality. This project builds an automated deep learning pipeline to detect and classify 8 different categories of tea leaf conditions with high precision.

By leveraging **Transfer Learning** with **ResNet50V2**, the model achieves a 94% F1-Score in identifying 8 different tea leaf conditions.

# 🍵 Tea Leaf Disease Classification using ResNet50V2

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Model](https://img.shields.io/badge/Model-ResNet50V2-red.svg)](#)
[![F1-Score](https://img.shields.io/badge/F1--Score-94%25-green.svg)](#)


## Dataset Analysis
The dataset contains 871 images categorized into 8 classes:
* **Diseases:** Anthracnose, Algal Leaf, Bird's Eye Spot, Brown Blight, Gray Blight, Red Leaf Spot, White Spot.
* **Healthy:** Healthy tea leaves.
* 
<img width="989" height="490" alt="crop1" src="https://github.com/user-attachments/assets/a9025150-b4c0-4060-b7f5-235dc22db20b" />

##  Results & Performance
The model performance is visualized below:

### Confusion Matrix

<img width="754" height="590" alt="matrix" src="https://github.com/user-attachments/assets/34a4d2c1-2b96-4ed6-a7e1-d7f4d3f882e7" />

##  Usage
Ensure your images are in the `dataset/` folder and run the notebook:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn

    
# Technical Workflow
1. Preprocessing

    Resizing: All images were resized to 224x224 pixels to match ResNet input requirements.

    Normalization: Pixel values were scaled using the ResNet50V2 specific preprocessing function.

2. Model Architecture

I utilized ResNet50V2 (Residual Network) due to its "identity shortcut connections," which help prevent the vanishing gradient problem in deep networks.

    Base Model: ResNet50V2 (Pre-trained on ImageNet).

    Global Average Pooling: Used to reduce spatial dimensions.

    Dense Layer: 256 neurons with ReLU activation.

    Dropout: Included to prevent overfitting.

    Output Layer: 8 neurons with Softmax activation.

3. Training Hyperparameters

    Optimizer: Adam (α=0.001).

    Loss Function: Categorical Crossentropy.

    Epochs: 20.

   # Results & Performance

The model achieved state-of-the-art results for this specific dataset:

    Overall Accuracy: 91%

    Macro F1-Score: 91%

    [!TIP]
    Key Insight: The model shows exceptional recall for the "Healthy" class, ensuring that farmers do not mistakenly treat healthy plants, thus saving costs and reducing chemical use.

 
