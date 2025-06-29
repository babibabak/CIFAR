# CIFAR
A CIFAR-10 image classification system built with Python and TensorFlow, utilizing deep and wide-and-deep neural networks to classify images into 10 categories. The project preprocesses the CIFAR-10 dataset, performs hyperparameter tuning with Keras Tuner, and evaluates models using accuracy, F1 score, and ROC AUC.
# CIFAR-10 Image Classification System
## Overview
This project implements an image classification system for the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes (e.g., airplane, automobile, bird). The system uses deep neural networks and wide-and-deep architectures built with TensorFlow, with hyperparameter tuning performed via Keras Tuner to optimize model performance. The models are evaluated using accuracy, F1 score, and ROC AUC metrics, achieving a test accuracy of approximately 44.5% and a ROC AUC of 85.4%. The project leverages Python with libraries like TensorFlow, Keras Tuner, Scikit-learn, NumPy, and Matplotlib for data preprocessing, model training, and evaluation.

## Features
- Data Preprocessing: Loads and normalizes the CIFAR-10 dataset, splitting it into training, validation, and test sets, and converts labels to one-hot encoded format.
- Model Architectures: Implements two models:
  - Deep Model: A simple neural network with a single hidden layer, tuned for the number of units (32–512).
  - Wide-and-Deep Model: Combines a wide layer and a deeper path with two hidden layers, tuned for unit counts.
- Hyperparameter Tuning: Uses Keras Tuner's RandomSearch to optimize the number of units in hidden layers, maximizing validation accuracy.
= Model Evaluation: Assesses performance using accuracy, weighted F1 score, and macro ROC AUC on the test set.
= Visualization: Supports plotting of training metrics (extendable for further analysis).

## Dataset
The CIFAR-10 dataset, included in TensorFlow/Keras, contains:
- Training Set: 50,000 images (split into 42,500 training and 7,500 validation images).
- Test Set: 10,000 images.
- Image Dimensions: 32x32 pixels with 3 color channels (RGB).
- Classes: 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- Labels: One-hot encoded for multi-class classification.

## Requirements
- Python 3.11
- Libraries: `tensorflow`, `keras-tuner`, `scikit-learn`, `numpy`, `matplotlib`

## Installation
1.Clone the repository:
```bash
git clone https://github.com/yourusername/cifar10-classification.git
```
2.Install dependencies:
```bash
pip install tensorflow keras-tuner scikit-learn numpy matplotlib
```
3.Run the Jupyter Notebook:
```bash
jupyter notebook CIFAR_10.ipynb
```

## Usage
- Load and preprocess the CIFAR-10 dataset, normalizing pixel values to [0, 1] and converting labels to one-hot encoded format.
= Perform hyperparameter tuning using Keras Tuner’s RandomSearch for both deep and wide-and-deep models.
= Train the models for 5 epochs with the Adam optimizer and categorical crossentropy loss.
= Evaluate the best models on the test set using accuracy, F1 score, and ROC AUC.
= Review the results summary for the best hyperparameters and model performance.

## Example Results
- Deep Model:
  - Test Accuracy: 44.49%
  - F1 Score: 43.68%
  - ROC AUC: 85.42%
  - Best Hyperparameter: 448 units in the hidden layer (validation accuracy: 43.27%).
- Wide-and-Deep Model:
  - Test Accuracy: 44.50%
  - F1 Score: 43.31%
  - ROC AUC: 85.48%
  - Best Hyperparameters: Tuned wide and deep units (validation accuracy: 43.29%).

## Methodology
- Data Preprocessing: Normalizes image pixel values to [0, 1] and splits the training set into 85% training and 15% validation. Labels are one-hot encoded for 10-class classification.
- Model Architecture:
  - Deep Model: Flattens the 32x32x3 input, followed by a single dense layer (tuned units: 32–512, ReLU activation) and a softmax output layer (10 classes).
  - Wide-and-Deep Model: Combines a wide dense layer and a two-layer deep path (tuned units: 32–512, ReLU activation), concatenated before a softmax output layer.
- Hyperparameter Tuning: Uses RandomSearch to test 5 trials (3 executions each) for optimal units in hidden layers, targeting maximum validation accuracy.
- Training: Models are trained for 5 epochs using the Adam optimizer and categorical crossentropy loss.
- Evaluation: Computes test accuracy, weighted F1 score, and macro ROC AUC for multi-class classification.

## Future Improvements
- Incorporate convolutional neural networks (CNNs) to better capture spatial features in images.
- Experiment with data augmentation (e.g., rotations, flips) to improve model robustness.
- Increase the number of epochs or trials for hyperparameter tuning to enhance performance.
- Add dropout or regularization to prevent overfitting.
- Visualize confusion matrices or class-specific performance metrics for deeper insights.

License
MIT License
