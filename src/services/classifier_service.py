"""
******************************
Author:
U3280553 U3243132 Assignment 3 Group 5 17/05/2026
Programming: Classification service. Trains a Random Forest classifier on
preprocessed image features, evaluates accuracy, saves the model and
evaluation outputs. Stage 2 component.
******************************
"""

"""
services/classifier_service.py
Handles Stage 2: training, evaluating, and saving the image classifier.

Uses a Random Forest from scikit-learn as the baseline model.
The preprocessed image features from ImagePreprocessor are used as input.

Outputs saved:
  - macro_classifier.joblib      : the trained model file
  - classification_report.txt    : precision, recall, F1 per class
  - confusion_matrix.png         : heatmap of predictions vs actual
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.config import (
    MODEL_FILE_NAME,
    MODEL_OUTPUT_DIR,
    N_ESTIMATORS,
    RANDOM_STATE,
    REPORT_OUTPUT_DIR,
    TEST_SIZE,
)
from src.services.image_preprocessor import ImagePreprocessor


class ClassifierService:
    """Train, evaluate, and persist the baseline image classification model.

    Responsibilities:
      - Convert image file paths into numeric feature arrays
      - Split data into training and test sets
      - Fit a Random Forest classifier
      - Produce evaluation metrics and save them to disk
      - Save and load the trained model with joblib
    """

    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        model_output_dir: Path = MODEL_OUTPUT_DIR,
        report_output_dir: Path = REPORT_OUTPUT_DIR,
    ) -> None:
        """Initialise the service with a preprocessor and output directories.

        Args:
            preprocessor:      An ImagePreprocessor instance.
            model_output_dir:  Where to save the trained model file.
            report_output_dir: Where to save evaluation report files.
        """
        self.preprocessor = preprocessor
        self.model_output_dir = model_output_dir
        self.report_output_dir = report_output_dir
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.report_output_dir.mkdir(parents=True, exist_ok=True)

        self.model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,  # use all available CPU cores
        )

    def prepare_features(
        self, dataframe: pd.DataFrame
    ) -> tuple:
        """Convert the indexed DataFrame into feature arrays for training.

        Args:
            dataframe: The indexed image DataFrame from DatasetIndexer.

        Returns:
            Tuple of (X, y) where X is a 2D NumPy array of features
            and y is a 1D array of string labels.
        """
        features = []
        labels = []
        total = len(dataframe)

        print(f"\nPreprocessing {total} images...")

        for i, (_, row) in enumerate(dataframe.iterrows(), start=1):
            try:
                feature_vector = self.preprocessor.transform(row["file_path"])
                features.append(feature_vector)
                labels.append(row["label"])
            except ValueError as error:
                print(f"  Skipping image {i}/{total}: {error}")

            if i % 200 == 0:
                print(f"  Processed {i}/{total} images...")

        print(f"  Done. {len(features)} images prepared.")
        return np.array(features), np.array(labels)

    def train(self, dataframe: pd.DataFrame) -> dict:
        """Fit the model on the dataset and return evaluation results.

        Args:
            dataframe: The indexed image DataFrame from DatasetIndexer.

        Returns:
            Dictionary containing accuracy, report string, confusion matrix,
            and class label list.
        """
        X, y = self.prepare_features(dataframe)

        print("\nSplitting into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        print(f"  Training samples : {len(X_train)}")
        print(f"  Test samples     : {len(X_test)}")

        print("\nTraining Random Forest classifier...")
        self.model.fit(X_train, y_train)
        print("  Training complete.")

        predictions = self.model.predict(X_test)
        labels = sorted(list(set(y)))

        results = {
            "accuracy": accuracy_score(y_test, predictions),
            "report": classification_report(y_test, predictions),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels),
            "labels": labels,
        }

        print(f"\n  Test accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
        return results

    def save_model(self) -> Path:
        """Save the trained model to disk using joblib.

        Returns:
            Path to the saved model file.
        """
        output_path = self.model_output_dir / MODEL_FILE_NAME
        joblib.dump(self.model, output_path)
        print(f"  Model saved: {output_path}")
        return output_path

    def load_model(self) -> None:
        """Load a previously saved model from disk."""
        model_path = self.model_output_dir / MODEL_FILE_NAME
        if not model_path.exists():
            raise FileNotFoundError(
                "No trained model found. Run option 3 (Train classifier) first."
            )
        self.model = joblib.load(model_path)

    def save_report(self, results: dict) -> None:
        """Write the classification report to a text file.

        Args:
            results: The results dict returned by train().
        """
        report_path = self.report_output_dir / "classification_report.txt"
        content = (
            f"Test Accuracy: {results['accuracy']:.4f} "
            f"({results['accuracy']*100:.1f}%)\n\n"
            f"{results['report']}"
        )
        report_path.write_text(content, encoding="utf-8")
        print(f"  Report saved: {report_path}")

    def save_confusion_matrix(self, results: dict) -> None:
        """Save a confusion matrix heatmap image.

        Args:
            results: The results dict returned by train().
        """
        labels = results["labels"]
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            results["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Confusion Matrix — Random Forest Classifier", pad=15)
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        output_path = self.report_output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix saved: {output_path}")
