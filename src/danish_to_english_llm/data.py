

from pathlib import Path
from datasets import load_dataset
import typer
from torch.utils.data import Dataset
import torch
import os
import random
class TranslationDataset(Dataset):
    """Dataset for Danish to English translation."""

    def __init__(self, mode: str) -> None:
        """Initialize dataset."""
        self.dataset = None
        self.data_path = "data/raw" 
        if mode not in ["train", "val", "test"]:
            raise ValueError("Invalid mode. Please choose from 'train', 'val', or 'test'.")
        self.mode = mode

        if len(os.listdir("data/raw")) == 1:
            self.download()
        if len(os.listdir("data/processed")) == 1:
            self.preprocess("data/processed")
        
        self.dataset = torch.load(f"data/processed/{mode}.pt", weights_only=True)
        

    def download(self) -> None:
        """Download dataset from HuggingFace."""
        self.dataset = load_dataset("kaitchup/opus-Danish-to-English")
        # Create raw data directory if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)
        self.dataset.save_to_disk(self.data_path)

    def preprocess(self, processed_data_path) -> None:
        """Preprocess the dataset by splitting on '###>'."""
        if self.dataset is None:
            self.dataset = load_dataset(str(self.data_path))
        
        self.processed_data_train = []
        for item in self.dataset['train']:
            text = item['text']
            if '###>' in text:
                danish, english = text.split('###>')
                self.processed_data_train.append({
                    'danish': danish.strip(),
                    'english': english.strip()
                })

        self.processed_data_val = []
        for item in self.dataset['validation']:
            text = item['text']
            if '###>' in text:
                danish, english = text.split('###>')
                self.processed_data_val.append({
                    'danish': danish.strip(),
                    'english': english.strip()
                })

        # Split the dataset into train and test sets
        processed_data_train_size = len(self.processed_data_train)
        train_size = int(0.9 * processed_data_train_size)

        random.shuffle(self.processed_data_train)
        self.train_data, self.test_data = self.processed_data_train[:train_size], self.processed_data_train[train_size:]
        
        # Save the splits
        torch.save(self.train_data, os.path.join(processed_data_path, "train.pt"))
        torch.save(self.test_data, os.path.join(processed_data_path, "test.pt"))
        torch.save(self.processed_data_val, os.path.join(processed_data_path, "val.pt"))

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        """Get item from dataset."""
        return self.dataset[idx]
    
if __name__ == "__main__":
    #typer.run(preprocess)
    dataset = TranslationDataset("train")
    c = 0
    for i in iter(dataset):
        print(i)
        c += 1
        if c == 10:
            break
