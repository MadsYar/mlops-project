import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import typer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import T5TokenizerFast


class TranslationDataset(Dataset):
    """Dataset for Danish to English translation using T5."""

    def __init__(self, mode: str, tokenizer: T5TokenizerFast, max_length: int = 128, data_dir: str = "data") -> None:
        """Initialize dataset."""
        if mode not in ["train", "val", "test"]:
            raise ValueError("Invalid mode. Please choose from 'train', 'val', or 'test'.")

        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_dir = Path(data_dir)

        self.data: List[Dict[str, str]] = []

        self.dataset = None

        # Create directories if they don't exist
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        try:
            # First check if processed data exists
            if any(self.processed_dir.glob("*.pt")):
                print(f"Loading processed data for {mode} split...")
                loaded_data = torch.load(self.processed_dir / f"{mode}.pt", weights_only=True)
                self.data = loaded_data
                print(f"Loaded {len(self.data)} examples for {mode} split")
                return

            # If no processed data, check raw data
            raw_files = list(self.raw_dir.glob("*"))
            only_gitkeep = len(raw_files) == 1 and raw_files[0].name == ".gitkeep"

            # Download if necessary
            if only_gitkeep:
                print("Raw directory empty. Downloading dataset...")
                self.dataset = self._download()
            else:
                print("Loading existing raw data...")
                self.dataset = load_dataset(str(self.raw_dir))

            # Preprocess
            print("Starting preprocessing...")
            self._preprocess()

            # Load the processed data after preprocessing
            print(f"Loading processed data for {mode} split...")
            loaded_data = torch.load(self.processed_dir / f"{mode}.pt", weights_only=True)
            self.data = loaded_data
            print(f"Loaded {len(self.data)} examples for {mode} split")

        except Exception as e:
            raise RuntimeError(f"Error initializing dataset: {str(e)}")

    def _download(self):
        """Download dataset from HuggingFace."""
        print("Downloading dataset from HuggingFace...")
        try:
            dataset = load_dataset("kaitchup/opus-Danish-to-English")
            dataset.save_to_disk(self.raw_dir)
            print("Dataset downloaded successfully")
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {str(e)}")

    def _preprocess(self) -> None:
        """Preprocess the dataset and create train/val/test splits."""
        print("Preprocessing dataset...")

        if self.dataset is None:
            raise ValueError("No dataset loaded. self.dataset is None.")

        try:
            # Process training data
            train_data = self._process_split(self.dataset["train"])
            val_data = self._process_split(self.dataset["validation"])

            print(f"Processed {len(train_data)} training examples")
            print(f"Processed {len(val_data)} validation examples")

            # Create train/test split from training data
            random.shuffle(train_data)
            split_idx = int(0.95 * len(train_data))
            final_train_data = train_data[:split_idx]
            test_data = train_data[split_idx:]

            # Save splits
            torch.save(final_train_data, self.processed_dir / "train.pt")
            torch.save(test_data, self.processed_dir / "test.pt")
            torch.save(val_data, self.processed_dir / "val.pt")

            print(
                f"Saved {len(final_train_data)} training, {len(test_data)} test, and {len(val_data)} validation examples"
            )

        except Exception as e:
            raise RuntimeError(f"Error during preprocessing: {str(e)}")

    def _process_split(self, split_data) -> List[Dict[str, str]]:
        """Process a single data split."""
        processed_data = []
        for item in split_data:
            text = item["text"]
            if "###>" in text:
                danish, english = text.split("###>")
                processed_data.append({"danish": danish.strip(), "english": english.strip()})
        return processed_data

    def _prepare_input(self, danish_text: str, english_text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and prepare input for the model."""
        # Prepare input
        source_encoding = self.tokenizer(
            f"translate Danish to English: {danish_text}",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Prepare target
        target_encoding = self.tokenizer(
            english_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": source_encoding.input_ids.squeeze(),
            "attention_mask": source_encoding.attention_mask.squeeze(),
            "labels": target_encoding.input_ids.squeeze(),
        }

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset."""
        item = self.data[idx]
        return self._prepare_input(item["danish"], item["english"])


def get_dataloaders(
    tokenizer: T5TokenizerFast, batch_size: int = 16, max_length: int = 128, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for training."""
    train_dataset = TranslationDataset("train", tokenizer, max_length)
    val_dataset = TranslationDataset("val", tokenizer, max_length)
    test_dataset = TranslationDataset("test", tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def preprocess(mode: str) -> TranslationDataset:
    """Preprocess the dataset."""
    dataset = TranslationDataset(mode, T5TokenizerFast.from_pretrained("google-t5/t5-small"))
    return dataset


if __name__ == "__main__":
    typer.run(preprocess)
