"""
******************************
Author:
U3280553 U3243132 Assignment 3 Group 5 17/05/2026
Programming: Image preprocessing service. Converts raw image files into
normalised grayscale feature vectors for use by the classifier. Stage 2 component.
******************************
"""

"""
services/image_preprocessor.py
Converts raw image files into normalised numeric feature arrays
that the classifier can use for training and prediction.

Design note: keeping preprocessing separate from the classifier means
we can change the preprocessing approach without touching model code,
and reuse the same preprocessor in both training and the console app.
"""

import cv2
import numpy as np

from src.config import IMAGE_SIZE


class ImagePreprocessor:
    """Convert raw images into model-ready numeric features.

    Each image is:
      1. Read from disk in grayscale (simplifies features, faster training)
      2. Resized to a fixed IMAGE_SIZE so all feature vectors are the same length
      3. Normalised to the 0–1 range (better for most classifiers)
      4. Flattened into a 1D array (required for scikit-learn models)
    """

    def __init__(self, image_size: tuple = IMAGE_SIZE) -> None:
        """Initialise the preprocessor with a target image size.

        Args:
            image_size: (width, height) tuple to resize all images to.
        """
        self.image_size = image_size

    def transform(self, file_path: str) -> np.ndarray:
        """Load, resize, normalise, and flatten one image.

        Args:
            file_path: Path to the image file as a string.

        Returns:
            A 1D NumPy array of float32 values in the range [0, 1].

        Raises:
            ValueError: If the image cannot be read from disk.
        """
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(
                f"Could not read image: {file_path}\n"
                "Check that the file path is correct and the file is not corrupted."
            )

        # Resize to the fixed target size
        resized = cv2.resize(image, self.image_size)

        # Normalise pixel values from 0-255 to 0.0-1.0
        normalised = resized.astype("float32") / 255.0

        # Flatten the 2D array into a 1D feature vector
        return normalised.flatten()
