"""
******************************
Author:
U3280553 U3243132 Assignment 3 Group 5 17/05/2026
Programming: Data class representing one indexed macroinvertebrate image.
Stores file path, species label, width, height, and channel count.
******************************
"""

"""
models/records.py
Data class representing one indexed macroinvertebrate image.
Using a dataclass here is an example of object-oriented design —
each image in the dataset becomes a structured ImageRecord object.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageRecord:
    """Store the core metadata for one indexed macroinvertebrate image.

    Attributes:
        file_path: Absolute path to the image file on disk.
        label:     Class name derived from the parent folder name.
        width:     Image width in pixels.
        height:    Image height in pixels.
        channels:  Number of colour channels (1 = grayscale, 3 = RGB).
    """

    file_path: Path
    label: str
    width: int
    height: int
    channels: int
