# Macroinvertebrate Image Analysis System
Unit: Software Technology 1 (4483/8995) — Assignment 3

## Project Goal
A modular Python application that indexes, analyses, and classifies
freshwater macroinvertebrate images from the Kaggle Stream Macroinvertebrates
dataset. It covers Stage 1 (EDA), Stage 2 (classification), and Stage 3
(console-based deployment).

## Main Features
- Dataset indexing with file path, label, and image dimensions
- Class distribution, image size, and sample grid visualisations (Stage 1)
- Random Forest image classifier with evaluation reports (Stage 2)
- Menu-driven console application for interactive use (Stage 3)
- Error handling for missing files, untrained models, and invalid input

## Python Packages Used
| Package | Purpose |
|---------|---------|
| `pathlib` | Clean file and folder path handling |
| `pandas` | Storing and analysing the image index as a DataFrame |
| `numpy` | Numerical array operations and feature preparation |
| `opencv-python` | Reading, resizing, and preprocessing images |
| `matplotlib` | Generating EDA and evaluation charts |
| `seaborn` | Styled statistical plots (class distribution, confusion matrix) |
| `scikit-learn` | Train/test split, Random Forest classifier, metrics |
| `joblib` | Saving and loading the trained model |
| `Pillow` | Image handling support |

## Installation
```bash
# 1. Clone or unzip the project
# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Dataset Setup
1. Download the dataset from:
   https://www.kaggle.com/datasets/kennethtm/stream-macroinvertebrates
2. Extract the zip file
3. Place the class folders inside `data/raw/`
   The structure should look like: `data/raw/ClassName/image.jpg`

## How to Run

### Option A — Full pipeline (non-interactive)
Runs Stage 1 EDA and Stage 2 training in one go:
```bash
python -m src.main
```

### Option B — Interactive console app (Stage 3)
Menu-driven application with all options:
```bash
python -m src.console_app
```
Menu options:
1. Show dataset summary
2. Generate EDA outputs (saved to `outputs/eda/`)
3. Train classifier (saved to `outputs/models/`)
4. Predict an image class
5. Exit

## Folder Structure
```
macro_project/
├── data/
│   └── raw/              ← Place Kaggle dataset folders here
├── outputs/
│   ├── eda/              ← EDA chart PNGs saved here
│   ├── models/           ← Trained model saved here
│   └── reports/          ← Classification report and confusion matrix
├── src/
│   ├── config.py         ← All paths and settings
│   ├── main.py           ← Non-interactive pipeline runner
│   ├── console_app.py    ← Stage 3 menu-driven console app
│   ├── models/
│   │   └── records.py    ← ImageRecord data class
│   └── services/
│       ├── dataset_indexer.py    ← Scans dataset, builds DataFrame
│       ├── eda_service.py        ← Generates Stage 1 EDA outputs
│       ├── image_preprocessor.py ← Converts images to model features
│       ├── classifier_service.py ← Trains and evaluates the model
│       └── workflow_service.py   ← Coordinates all stages
├── MANUAL_TESTING.md     ← Testing evidence
├── README.md
└── requirements.txt
```

## Work Division
| Member               | Responsibility |
|----------------------|---------------|
| [liam Walther]       | Dataset indexing, EDA service, Stage 1 outputs |
| [Tom]                | Image preprocessing, classifier service, evaluation outputs |
| [Liam Walther & Tom] | Console app, testing, README, integration |

## Reused / Adapted Code
Code in this project is adapted from course-provided guidance materials
(Assignment 3 Full Guidance and Coding Examples document, Software Technology 1).
All adaptations are our own implementation of the suggested structure.
