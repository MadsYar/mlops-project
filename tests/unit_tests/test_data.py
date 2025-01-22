import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from danish_to_english_llm.data import TranslationDataset, get_dataloaders
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock(spec=T5TokenizerFast)

    # Mock tokenizer output structure
    encoding = Mock()
    encoding.input_ids = torch.tensor([[1, 2, 3]])
    encoding.attention_mask = torch.tensor([[1, 1, 1]])

    # Mock the call behavior
    tokenizer.return_value = encoding
    tokenizer.__call__ = Mock(return_value=encoding)

    return tokenizer


@pytest.fixture
def sample_processed_data():
    return [{"danish": "Hej verden", "english": "Hello world"}, {"danish": "God morgen", "english": "Good morning"}]


@pytest.fixture
def mock_dataset():
    dataset = {
        "train": [{"text": "Hej verden ###> Hello world"}],
        "validation": [{"text": "God morgen ###> Good morning"}],
    }
    return dataset


def test_translation_dataset_initialization(mock_tokenizer, tmp_path):
    """Test successful initialization of dataset."""
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)

    # Create dummy processed data
    torch.save([{"danish": "test", "english": "test"}], processed_dir / "train.pt")

    dataset = TranslationDataset("train", mock_tokenizer, data_dir=str(data_dir))
    assert len(dataset) == 1
    assert isinstance(dataset, TranslationDataset)


@patch("danish_to_english_llm.data.load_dataset")
def test_translation_dataset_download(mock_load_dataset, mock_dataset, mock_tokenizer, tmp_path):
    """Test dataset download functionality."""
    mock_load_dataset.return_value = mock_dataset

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    dataset = TranslationDataset("train", mock_tokenizer, data_dir=str(data_dir))

    assert mock_load_dataset.called
    assert isinstance(dataset, TranslationDataset)


def test_translation_dataset_getitem(mock_tokenizer, tmp_path, sample_processed_data):
    """Test __getitem__ functionality."""
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)

    torch.save(sample_processed_data, processed_dir / "train.pt")
    dataset = TranslationDataset("train", mock_tokenizer, data_dir=str(data_dir))

    # Mock the _prepare_input method
    dataset._prepare_input = Mock(
        return_value={
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([4, 5, 6]),
        }
    )

    item = dataset[0]
    assert isinstance(item, dict)
    assert all(k in item for k in ["input_ids", "attention_mask", "labels"])


def test_get_dataloaders(mock_tokenizer, tmp_path):
    """Test dataloader creation."""
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)

    # Create dummy processed data for all splits
    dummy_data = [{"danish": "test", "english": "test"}]
    for split in ["train", "val", "test"]:
        torch.save(dummy_data, processed_dir / f"{split}.pt")

    with patch("danish_to_english_llm.data.TranslationDataset") as MockDataset:
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        MockDataset.return_value = mock_dataset

        train_loader, val_loader, test_loader = get_dataloaders(mock_tokenizer, batch_size=2, num_workers=0)

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)


@patch("danish_to_english_llm.data.load_dataset")
def test_process_split(mock_load_dataset, mock_tokenizer):
    """Test _process_split method."""

    mock_load_dataset.return_value = {
        "train": [{"text": "dansk ###> english"}, {"text": "mere dansk ###> more english"}],
        "validation": [{"text": "val dansk ###> val english"}],
    }

    dataset = TranslationDataset("train", mock_tokenizer)
    test_data = [{"text": "dansk ###> english"}, {"text": "mere dansk ###> more english"}]

    processed = dataset._process_split(test_data)
    assert len(processed) == 2
    assert all(isinstance(item, dict) for item in processed)
    assert all("danish" in item and "english" in item for item in processed)


@pytest.mark.parametrize("invalid_mode", ["invalid", "testing", ""])
def test_invalid_mode(invalid_mode, mock_tokenizer):
    """Test that invalid modes raise ValueError."""
    with pytest.raises(ValueError):
        TranslationDataset(invalid_mode, mock_tokenizer)


@patch("danish_to_english_llm.data.load_dataset")
def test_prepare_input(mock_load_dataset, mock_tokenizer):
    """Test _prepare_input method."""

    mock_load_dataset.return_value = {
        "train": [{"text": "dansk ###> english"}, {"text": "mere dansk ###> more english"}],
        "validation": [{"text": "val dansk ###> val english"}],
    }

    dataset = TranslationDataset("train", mock_tokenizer)

    # Mock tokenizer output for this specific test
    mock_encoding = Mock()
    mock_encoding.input_ids = torch.tensor([[1, 2, 3]])
    mock_encoding.attention_mask = torch.tensor([[1, 1, 1]])
    mock_tokenizer.__call__.return_value = mock_encoding

    result = dataset._prepare_input("dansk tekst", "english text")
    assert isinstance(result, dict)
    assert all(k in result for k in ["input_ids", "attention_mask", "labels"])


def test_dataset_length(mock_tokenizer, tmp_path, sample_processed_data):
    """Test __len__ method."""
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)

    torch.save(sample_processed_data, processed_dir / "train.pt")

    dataset = TranslationDataset("train", mock_tokenizer, data_dir=str(data_dir))
    assert len(dataset) == len(sample_processed_data)
