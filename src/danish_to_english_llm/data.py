

from pathlib import Path
from datasets import load_dataset
import typer
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Dataset for Danish to English translation."""

    def __init__(self) -> None:
        # self.data_path = raw_data_path
        self.dataset = load_dataset("kaitchup/opus-Danish-to-English")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset['train'])

    def __getitem__(self, index: int):
        """Return a translation pair from the dataset."""
        item = self.dataset['train'][index]
        return {
            'danish': item['da'],
            'english': item['en']
        }

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)
        # Save the dataset splits if needed
        self.dataset.save_to_disk(output_folder)


def preprocess(output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset()
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)

# %%
