"""
******************************
Author:
U3280553 U3243132 Assignment 3 Group 5 17/05/2026
Programming: Workflow coordinator service. Coordinates all shared project
actions used by both main.py and console_app.py. Owns one instance of
each service class.
******************************
"""

"""
services/workflow_service.py
Coordinates all shared project actions used by both main.py and console_app.py.

Having one WorkflowService means neither the batch script nor the console app
duplicate any logic — they both call the same methods here.
"""

from pathlib import Path

import pandas as pd

from src.config import EDA_OUTPUT_DIR, MODEL_OUTPUT_DIR, MODEL_FILE_NAME
from src.services.classifier_service import ClassifierService
from src.services.dataset_indexer import DatasetIndexer
from src.services.eda_service import EDAService
from src.services.image_preprocessor import ImagePreprocessor


class WorkflowService:
    """Coordinate the shared workflow used by batch and console entry points.

    This is the central hub of the application. It owns instances of the
    other service classes and exposes high-level methods for each stage.
    """

    def __init__(self) -> None:
        """Initialise all service instances and ensure output folders exist."""
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.indexer = DatasetIndexer()
        self.preprocessor = ImagePreprocessor()
        self.classifier = ClassifierService(self.preprocessor)
        self._dataframe: pd.DataFrame | None = None

    def load_dataframe(self) -> pd.DataFrame:
        """Load and cache the indexed dataset (only scans disk once).

        Returns:
            The indexed image DataFrame.
        """
        if self._dataframe is None:
            print("Scanning dataset folder...")
            self._dataframe = self.indexer.build_dataframe()
            print(f"Found {len(self._dataframe)} images.")
        return self._dataframe

    def show_summary(self) -> dict:
        """Print dataset summary statistics to the console.

        Returns:
            Dictionary of summary statistics.
        """
        dataframe = self.load_dataframe()
        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        return eda.print_summary()

    def generate_eda(self) -> None:
        """Generate and save all EDA chart outputs."""
        dataframe = self.load_dataframe()
        eda = EDAService(dataframe, EDA_OUTPUT_DIR)
        eda.run_all()

    def train_model(self) -> dict:
        """Train the classifier and save the model and evaluation outputs.

        Returns:
            The results dictionary from ClassifierService.train().
        """
        dataframe = self.load_dataframe()
        print("\n--- Stage 2: Training Classifier ---")
        results = self.classifier.train(dataframe)
        self.classifier.save_model()
        self.classifier.save_report(results)
        self.classifier.save_confusion_matrix(results)
        print("\nTraining complete. All outputs saved.")
        return results

    def predict_image(self, file_path: str) -> str:
        """Predict the macroinvertebrate class for a single image.

        Args:
            file_path: Path to the image file to classify.

        Returns:
            The predicted class name as a string.

        Raises:
            FileNotFoundError: If the model has not been trained yet.
            ValueError: If the image cannot be read.
        """
        # Load saved model if not already in memory
        model_path = MODEL_OUTPUT_DIR / MODEL_FILE_NAME
        if not model_path.exists():
            raise FileNotFoundError(
                "No trained model found. Please run option 3 (Train classifier) first."
            )
        self.classifier.load_model()

        features = self.preprocessor.transform(file_path).reshape(1, -1)
        prediction = self.classifier.model.predict(features)[0]

        if hasattr(self.classifier.model, "predict_proba"):
            probability = self.classifier.model.predict_proba(features).max()
            print(f"\n  Predicted class : {prediction}")
            print(f"  Confidence      : {probability:.1%}")
        else:
            print(f"\n  Predicted class : {prediction}")

        return str(prediction)

    def run_full_pipeline(self) -> None:
        """Run the complete Stage 1 and Stage 2 pipeline non-interactively."""
        print("=" * 50)
        print("  Macroinvertebrate Image Analysis System")
        print("  Full Pipeline Run")
        print("=" * 50)
        self.generate_eda()
        results = self.train_model()
        print(f"\nFinal test accuracy: {results['accuracy']:.4f}")
