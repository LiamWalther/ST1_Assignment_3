"""
******************************
Author:
U3280553 U3243132 Assignment 3 Group 5 17/05/2026
Programming: Central configuration module. Defines all project paths and
shared settings used across every other module.
******************************
"""

"""
config.py
Central configuration for the Macroinvertebrate Image Analysis System.
All paths and shared settings live here so nothing is hard-coded elsewhere.
"""

from pathlib import Path

# Root of the project (two levels up from this file: src/ -> macro_project/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
MODEL_OUTPUT_DIR = OUTPUTS_DIR / "models"
REPORT_OUTPUT_DIR = OUTPUTS_DIR / "reports"

# Image settings
IMAGE_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Model settings
MODEL_FILE_NAME = "macro_classifier.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
