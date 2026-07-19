"""
DecodeLabs - Project 2: Data Classification Using AI
Author: Ammara Manzoor
Description: A basic supervised learning classification model that
             loads a dataset, splits it into training/testing sets,
             trains a classifier, and evaluates its accuracy.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


def load_dataset():
    """Loads the Iris dataset and returns features (X) and labels (y)."""
    iris = load_iris()
    X = iris.data          # features: sepal/petal length & width
    y = iris.target        # labels: flower species (0, 1, 2)
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names


def explore_dataset(X, y, feature_names, target_names):
    """Prints a basic overview of the dataset to understand its structure."""
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [target_names[label] for label in y]

    print("Dataset Preview:")
    print(df.head(), "\n")

    print("Dataset Shape:", X.shape)
    print("Classes:", list(target_names))
    print("Samples per class:", dict(zip(*pd_unique(y, target_names))), "\n")


def pd_unique(y, target_names):
    """Helper function to count samples per class."""
    import numpy as np
    unique, counts = np.unique(y, return_counts=True)
    labels = [target_names[i] for i in unique]
    return labels, counts


def train_model(X_train, y_train):
    """Trains a K-Nearest Neighbors classifier on the training data."""
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluates the trained model on the test set and prints results."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model Evaluation")
    print("-" * 40)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))


def predict_new_sample(model, target_names):
    """Demonstrates prediction on a new, unseen data sample."""
    # Example: sepal length, sepal width, petal length, petal width
    new_sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(new_sample)
    predicted_class = target_names[prediction[0]]

    print("New Sample Prediction")
    print("-" * 40)
    print(f"Input: {new_sample[0]}")
    print(f"Predicted Class: {predicted_class}")


def main():
    # Step 1: Load and understand the dataset
    X, y, feature_names, target_names = load_dataset()
    explore_dataset(X, y, feature_names, target_names)

    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")

    # Step 3: Apply a simple classification algorithm
    model = train_model(X_train, y_train)

    # Step 4: Evaluate the model
    evaluate_model(model, X_test, y_test, target_names)

    # Step 5: Predict on new/unseen data
    predict_new_sample(model, target_names)


if __name__ == "__main__":
    main()
