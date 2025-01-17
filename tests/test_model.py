import pytest
import torch
from danish_to_english_llm.model import T5LightningModel


@pytest.fixture(scope="module")
def model():
    """Creating a model for all tests in this module.."""
    # Use a small model for faster tests
    model_instance = T5LightningModel(pretrained_model="t5-small")
    return model_instance


def test_model_initialization(model):
    """Testing if the model has been load correctly."""
    assert model is not None
    assert model.model is not None
    assert model.tokenizer is not None
    assert model.learning_rate == 1e-4
    assert model.max_length == 128
    # Ensure that train_losses and val_losses lists are initialized
    assert isinstance(model.train_losses, list)
    assert isinstance(model.val_losses, list)


def test_forward_pass(model):
    """Test the forward pass."""
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])
    labels = torch.tensor([[4, 5, 6]])

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    # Check that the outputs contain a loss (training scenario) and logits
    assert hasattr(outputs, "loss")
    assert hasattr(outputs, "logits")
    assert isinstance(outputs.loss, torch.Tensor)
    assert isinstance(outputs.logits, torch.Tensor)


def test_training_step(model):
    """Test the training step to ensure it returns a loss tensor."""
    # Clear any previous training losses
    model.train_losses = []

    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[4, 5, 6]]),
    }

    loss = model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor), "training_step must return a torch.Tensor"
    assert len(model.train_losses) == 1, "Loss should be appended to train_losses list"
    assert not torch.isnan(loss).any().item(), "Loss should not be NaN"


def test_validation_step(model):
    """Test the validation step to ensure it logs a loss to val_losses."""
    # Clear any previous validation losses
    model.val_losses = []

    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[4, 5, 6]]),
    }

    model.validation_step(batch, batch_idx=0)
    assert len(model.val_losses) == 1, "Loss should be appended to val_losses list"


def test_configure_optimizers(model):
    """Test that configure_optimizers returns the expected dictionary with an optimizer and a scheduler."""
    # We mock out the trainer attributes needed by OneCycleLR
    model.trainer = type("mock_trainer", (), {"estimated_stepping_batches": 100, "max_epochs": 1})()

    opt_dict = model.configure_optimizers()
    assert "optimizer" in opt_dict, "Should return an optimizer"
    assert "lr_scheduler" in opt_dict, "Should return a lr_scheduler dict"
    optimizer = opt_dict["optimizer"]
    lr_scheduler = opt_dict["lr_scheduler"]["scheduler"]
    assert isinstance(optimizer, torch.optim.AdamW), "Optimizer should be AdamW"
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR), "Scheduler should be OneCycleLR"


def test_translate(model):
    """Test the translate function to ensure it produces a string output."""
    danish_text = "Hej, hvordan har du det?"
    translation = model.translate(danish_text)
    assert isinstance(translation, str)
    assert len(translation) > 0, "Translation should not be empty"


def test_get_metrics(model):
    """Test that get_metrics returns a dictionary with train and val losses."""
    # Just ensure it returns the logs so far
    metrics = model.get_metrics()
    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert isinstance(metrics["train_loss"], list)
    assert isinstance(metrics["val_loss"], list)
