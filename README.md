# decodelabs-data-classification
# Data Classification Using AI

**DecodeLabs — Project 2: Predictive Phase**

A supervised learning classification model built in Python using scikit-learn, demonstrating the fundamental machine learning pipeline — loading data, splitting it into training and testing sets, training a classifier, and validating its performance.

---

## Project Overview

This project moves beyond simple rule-based logic into supervised learning. Using the Iris dataset, a classic benchmark dataset in machine learning, the model learns to recognize patterns in flower measurements and classify new, unseen samples into one of three species. The goal is to understand and implement the core classification pipeline: data loading, preprocessing, training, and evaluation.

---

## Features

- Loads and explores a real-world dataset (Iris)
- Displays dataset structure, class distribution, and sample preview
- Splits data into training (80%) and testing (20%) sets
- Trains a K-Nearest Neighbors (KNN) classification model
- Evaluates model performance using accuracy and a detailed classification report
- Predicts the class of a new, unseen data sample

---

## Tech Stack

| Component  | Details                          |
|------------|-----------------------------------|
| Language   | Python 3                         |
| Libraries  | scikit-learn, pandas, numpy      |
| Algorithm  | K-Nearest Neighbors (KNN)        |
| Dataset    | Iris Dataset (150 samples, 3 classes) |

---

## Project Structure

├── classifier.py     # Main classification model
└── README.md          # Project documentation

---

## How to Run

1. Ensure Python 3 is installed on your system.
2. Install the required libraries:

pip install scikit-learn pandas

3. Run the following command in your terminal:

python classifier.py

---

## Dataset Information

The Iris dataset contains 150 samples of iris flowers, evenly split across three species: setosa, versicolor, and virginica. Each sample includes four features: sepal length, sepal width, petal length, and petal width.

---

## Model Pipeline

1. Load Dataset: Import the Iris dataset and extract features and labels.
2. Explore Data: Preview the dataset and understand its structure and class distribution.
3. Split Data: Divide the dataset into 80 percent training data and 20 percent testing data.
4. Train Model: Fit a K-Nearest Neighbors classifier on the training data.
5. Evaluate Model: Test the model on unseen data and measure accuracy, precision, recall, and F1-score.
6. Predict New Sample: Use the trained model to classify a new, unseen flower measurement.

---

## Sample Output

Dataset Shape: (150, 4)
Classes: setosa, versicolor, virginica
Samples per class: 50 each

Training samples: 120
Testing samples: 30

Model Evaluation
Accuracy: 100.00%

Classification Report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

New Sample Prediction
Input: [5.1, 3.5, 1.4, 0.2]
Predicted Class: setosa

---

## Key Concepts Demonstrated

- Data Handling: Loading, structuring, and exploring a dataset
- Supervised Learning Basics: Training a model on labeled data
- Model Training and Validation: Splitting data and measuring performance
- Classification Algorithms: Applying K-Nearest Neighbors for pattern recognition

---

## Future Enhancements

- Experiment with additional classification algorithms such as Decision Trees or Support Vector Machines
- Apply the pipeline to larger, more complex datasets
- Implement cross-validation for more robust model evaluation
- Add data visualization to illustrate class separability
