"""
******************************
Author:
U3280553 U3243132 Assignment 3 Group 5 17/05/2026
Programming: Exploratory Data Analysis service. Generates and saves class
distribution charts, image size histograms, and a sample image grid.
Stage 1 component.
******************************
"""

"""
services/eda_service.py
Generates and saves all Stage 1 Exploratory Data Analysis outputs.

Outputs produced:
  - class_distribution.png    : bar chart of images per class
  - image_size_distribution.png : width and height histograms
  - sample_grid.png           : 3x3 grid of representative images
  - summary statistics        : returned as a dict and printed to console
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class EDAService:
    """Generate and save EDA outputs for the indexed image dataset.

    All chart files are written to output_dir so they can be shown
    during the Week 13 presentation and included in the report.
    """

    def __init__(self, dataframe: pd.DataFrame, output_dir: Path) -> None:
        """Initialise with the indexed dataset and target output folder.

        Args:
            dataframe:  The indexed image DataFrame from DatasetIndexer.
            output_dir: Folder where chart images will be saved.
        """
        self.dataframe = dataframe
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_class_distribution(self) -> None:
        """Save a bar chart showing how many images exist per class."""
        plt.figure(figsize=(14, 6))
        order = self.dataframe["label"].value_counts().index
        sns.countplot(data=self.dataframe, x="label", order=order, palette="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.title("Macroinvertebrate Images per Class", fontsize=14, pad=15)
        plt.xlabel("Class")
        plt.ylabel("Image Count")
        plt.tight_layout()
        plt.savefig(self.output_dir / "class_distribution.png", dpi=150)
        plt.close()
        print("  Saved: class_distribution.png")

    def save_image_size_distribution(self) -> None:
        """Save histograms showing the distribution of image widths and heights."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.histplot(self.dataframe["width"], bins=20, ax=axes[0], color="steelblue")
        sns.histplot(self.dataframe["height"], bins=20, ax=axes[1], color="darkorange")
        axes[0].set_title("Image Width Distribution")
        axes[1].set_title("Image Height Distribution")
        axes[0].set_xlabel("Width (px)")
        axes[1].set_xlabel("Height (px)")
        plt.suptitle("Image Dimension Distributions", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / "image_size_distribution.png", dpi=150)
        plt.close()
        print("  Saved: image_size_distribution.png")

    def save_sample_grid(self, sample_count: int = 9) -> None:
        """Save a 3x3 grid of randomly sampled images for visual inspection.

        Args:
            sample_count: Number of images to show (max 9 for a 3x3 grid).
        """
        sample_df = self.dataframe.sample(
            min(sample_count, len(self.dataframe)), random_state=42
        )
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))

        for ax, (_, row) in zip(axes.flat, sample_df.iterrows()):
            image = cv2.imread(row["file_path"])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax.imshow(image)
            ax.set_title(row["label"], fontsize=8)
            ax.axis("off")

        # Hide any unused grid cells
        for ax in axes.flat[len(sample_df):]:
            ax.axis("off")

        plt.suptitle("Sample Images from Dataset", fontsize=13)
        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_grid.png", dpi=150)
        plt.close()
        print("  Saved: sample_grid.png")

    def build_summary(self) -> dict:
        """Calculate and return key dataset statistics.

        Returns:
            Dictionary with total_images, total_classes, mean_width,
            mean_height, and class_counts.
        """
        summary = {
            "total_images": int(len(self.dataframe)),
            "total_classes": int(self.dataframe["label"].nunique()),
            "mean_width": round(float(self.dataframe["width"].mean()), 1),
            "mean_height": round(float(self.dataframe["height"].mean()), 1),
            "min_images_per_class": int(
                self.dataframe["label"].value_counts().min()
            ),
            "max_images_per_class": int(
                self.dataframe["label"].value_counts().max()
            ),
        }
        return summary

    def print_summary(self) -> dict:
        """Print a formatted dataset summary to the console and return it."""
        summary = self.build_summary()
        print("\n" + "=" * 45)
        print("  DATASET SUMMARY")
        print("=" * 45)
        print(f"  Total images      : {summary['total_images']}")
        print(f"  Total classes     : {summary['total_classes']}")
        print(f"  Mean width        : {summary['mean_width']} px")
        print(f"  Mean height       : {summary['mean_height']} px")
        print(f"  Min images/class  : {summary['min_images_per_class']}")
        print(f"  Max images/class  : {summary['max_images_per_class']}")
        print("=" * 45)

        print("\n  Images per class:")
        class_counts = self.dataframe["label"].value_counts()
        for label, count in class_counts.items():
            print(f"    {label:<30} {count}")

        return summary

    def run_all(self) -> dict:
        """Run every EDA output in one call. Returns the summary dict."""
        print("\nGenerating EDA outputs...")
        self.save_class_distribution()
        self.save_image_size_distribution()
        self.save_sample_grid()
        summary = self.print_summary()
        print(f"\nAll EDA outputs saved to: {self.output_dir}")
        return summary
