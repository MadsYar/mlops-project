from danish_to_english_llm.data import TranslationDataset
from transformers import T5TokenizerFast
from torch.utils.data import Dataset
import os
import pytest

@pytest.mark.skipif(not os.path.exists('data/raw'), reason="Data files not found")
def test_translation_dataset():
    """Test the TranslationDataset class."""
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    dataset = TranslationDataset("train", tokenizer)
    assert isinstance(dataset, Dataset)
