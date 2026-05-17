"""
******************************
Author:
U3280553 U3243132 Assignment 3 Group 5 17/05/2026
Programming: Dataset indexing service. Scans the raw image folder recursively,
reads image dimensions with OpenCV, and builds a Pandas DataFrame with one
row per image. Stage 1 component.
******************************
"""

"""
services/dataset_indexer.py
Scans the raw dataset folder and builds a Pandas DataFrame where
each row represents one image with its file path, label, and dimensions.

This is Stage 1 work. The DataFrame produced here is reused in the
EDA service and the classifier service, which is why separating it
into its own class is good design.
"""

from pathlib import Path

import cv2
import pandas as pd

from src.config import RAW_DATA_DIR, SUPPORTED_EXTENSIONS


class DatasetIndexer:
    """Scan the dataset folder and build a tabular image index.

    The label for each image is taken from the name of its parent folder,
    which matches the structure of the Kaggle macroinvertebrate dataset.
    """

    def __init__(self, data_dir: Path = RAW_DATA_DIR) -> None:
        """Initialise the indexer with the path to the raw data folder.

        Args:
            data_dir: Path to the top-level folder containing class subfolders.
        """
        self.data_dir = data_dir

    def build_dataframe(self) -> pd.DataFrame:
        """Scan all image files recursively and return one row per image.

        Returns:
            A Pandas DataFrame with columns:
            file_path, label, width, height, channels.

        Raises:
            FileNotFoundError: If the data directory does not exist.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {self.data_dir}\n"
                "Please place the Kaggle dataset inside data/raw/"
            )

        records = []

        for file_path in self.data_dir.rglob("*"):
            # Skip anything that is not a supported image format
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # Try to read the image with OpenCV
            image = cv2.imread(str(file_path))
            if image is None:
                continue

            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1

            # The class label is the name of the immediate parent folder
            label = file_path.parent.name

            records.append(
                {
                    "file_path": str(file_path),
                    "label": label,
                    "width": width,
                    "height": height,
                    "channels": channels,
                }
            )

        if not records:
            raise ValueError(
                "No valid images were found in the dataset folder. "
                "Check that the Kaggle dataset is extracted into data/raw/"
            )

        return pd.DataFrame(records)
