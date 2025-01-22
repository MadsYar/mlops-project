from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger


def dataset_statistics(data_dir: str = "data") -> Dict:
    """
    Generate and visualize statistics for the translation dataset.

    Args:
        data_dir: Root directory containing the dataset files

    Returns:
        Dict containing various statistics about the dataset
    """
    logger.info("Calculating dataset statistics...")
    data_path = Path(data_dir) / "processed"

    if not data_path.exists():
        raise ValueError(f"Dataset directory {data_path} does not exist")

    # Load the datasets
    try:
        train_data = torch.load(data_path / "train.pt", weights_only=True)
        val_data = torch.load(data_path / "val.pt", weights_only=True)
        test_data = torch.load(data_path / "test.pt", weights_only=True)
    except Exception as e:
        logger.error(f"Error loading dataset files: {str(e)}")
        raise RuntimeError(f"Error loading dataset files: {str(e)}")

    # Calculate basic statistics
    stats = {
        "num_train_samples": len(train_data),
        "num_val_samples": len(val_data),
        "num_test_samples": len(test_data),
        "total_samples": len(train_data) + len(val_data) + len(test_data),
    }

    # Calculate length statistics
    length_stats = calculate_length_statistics(train_data, val_data, test_data)
    stats.update(length_stats)

    # Generate visualizations
    generate_visualizations(train_data, val_data, test_data, data_dir)

    # Print summary
    logger.info("\nDataset Statistics Summary:")
    logger.info(f"Total number of samples: {stats['total_samples']}")
    logger.info(f"Training samples: {stats['num_train_samples']}")
    logger.info(f"Validation samples: {stats['num_val_samples']}")
    logger.info(f"Test samples: {stats['num_test_samples']}")
    logger.info(f"\nAverage sentence lengths:")
    logger.info(f"Danish: {stats['avg_danish_length']:.2f} characters")
    logger.info(f"English: {stats['avg_english_length']:.2f} characters")

    return stats


def calculate_length_statistics(train_data, val_data, test_data) -> Dict:
    """Calculate various length-based statistics for the dataset."""
    all_data = train_data + val_data + test_data

    danish_lengths = [len(item["danish"]) for item in all_data]
    english_lengths = [len(item["english"]) for item in all_data]

    return {
        "avg_danish_length": np.mean(danish_lengths),
        "avg_english_length": np.mean(english_lengths),
        "max_danish_length": max(danish_lengths),
        "max_english_length": max(english_lengths),
        "min_danish_length": min(danish_lengths),
        "min_english_length": min(english_lengths),
    }


def generate_visualizations(train_data, val_data, test_data, data_dir: str) -> None:
    """Generate visualization plots for the dataset."""
    output_dir = Path(data_dir) / "stats"
    output_dir.mkdir(exist_ok=True)

    # Prepare data for plotting
    all_data = train_data + val_data + test_data
    danish_lengths = [len(item["danish"]) for item in all_data]
    english_lengths = [len(item["english"]) for item in all_data]

    # Plot 1: Distribution of sentence lengths
    plt.figure(figsize=(12, 6))
    plt.hist(danish_lengths, alpha=0.5, label="Danish", bins=50)
    plt.hist(english_lengths, alpha=0.5, label="English", bins=50)
    plt.xlabel("Sentence Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Sentence Lengths")
    plt.legend()
    plt.savefig(output_dir / "sentence_length_distribution.png")
    plt.close()

    # Plot 2: Split distribution
    plt.figure(figsize=(8, 6))
    splits = ["Train", "Validation", "Test"]
    sizes = [len(train_data), len(val_data), len(test_data)]
    plt.pie(sizes, labels=splits, autopct="%1.1f%%")
    plt.title("Dataset Split Distribution")
    plt.savefig(output_dir / "split_distribution.png")
    plt.close()

    # Plot 3: Length correlation
    plt.figure(figsize=(10, 6))
    plt.scatter(danish_lengths, english_lengths, alpha=0.1)
    plt.xlabel("Danish Sentence Length")
    plt.ylabel("English Sentence Length")
    plt.title("Correlation between Danish and English Sentence Lengths")
    plt.savefig(output_dir / "length_correlation.png")
    plt.close()

    logger.info(f"Visualizations saved to {output_dir}")
