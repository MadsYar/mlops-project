from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from loguru import logger
from transformers import T5ForConditionalGeneration, T5TokenizerFast

logger.add(
    "logs/data.log",
    rotation="100 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} {level} [{file.name}:{line}] {message}",
)


class T5LightningModel(pl.LightningModule):
    """
    Implementation of a T5 model for danish-english translation using PyTorch Lightning.

        Attributes:
            learning_rate (float): Learning rate for the optimizer.
            max_length (int): Maximum length for inputs and outputs.
            model (T5ForConditionalGeneration): The underlying T5 model.
            tokenizer (T5TokenizerFast): T5 tokenizer for processing text.
            train_losses (List[float]): List of training losses.
            val_losses (List[float]): List of validation losses.

    """

    def __init__(
        self,
        pretrained_model: str = "google-t5/t5-small",
        learning_rate: float = 1e-4,
        max_length: int = 128,
    ) -> None:
        """Initialize the T5 Lightning model.

        Args:
            pretrained_model: Path for the pretrained T5 model.
            learning_rate: Learning rate for the optimizer.
            max_length: Maximum sequence length for tokenization.
        """

        super().__init__()
        self.learning_rate = learning_rate
        self.max_length = max_length

        # Load pretrained model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.tokenizer = T5TokenizerFast.from_pretrained(pretrained_model)

        # Initialize metrics
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.

            Args:
                input_ids: Tensor of input token IDs.
                attention_mask: Tensor indicating which tokens should be attended to.
                labels: Optional tensor of target token IDs for training.

            Returns:
                torch.Tensor: Model outputs containing loss and logits.
        """

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Performs a training step.

            Args:
                batch: Dictionary containing input_ids, attention_mask, and labels.
                batch_idx: Index of the current batch.

            Returns:
                torch.Tensor: Computed loss for backpropagation.
        """

        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])

        loss = outputs.loss
        self.train_losses.append(loss.item())

        # Log training loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Performs a validation step.

            Args:
                batch: Dictionary containing input_ids, attention_mask, and labels.
                batch_idx: Index of the current batch.
        """

        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])

        loss = outputs.loss
        self.val_losses.append(loss.item())

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

            Returns:
                Dict: Configuration dictionary containing optimizer (AdamW) and scheduler (OneCycleLR).
        """

        # Prepare optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Add linear scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=self.trainer.estimated_stepping_batches,
            epochs=self.trainer.max_epochs,
            pct_start=0.1,
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def get_metrics(self):
        """Get training metrics."""

        return {"train_loss": self.train_losses, "val_loss": self.val_losses}

    def translate(self, danish_text: str) -> str:
        """
        Translates Danish text to English.

            Args:
                danish_text: Input text in Danish

            Returns:
                str: Translated text in English
        """

        # Prepare input
        inputs = self.tokenizer(
            f"translate Danish to English: {danish_text}",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate translation
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.max_length,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )

        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example usage
    logger.info("Initializing model")
    model = T5LightningModel()

    # If you have a trained model, load it:
    try:
        model.load_state_dict(torch.load("models/final_model.pth", weights_only=True))
        model.eval()  # Set to evaluation mode
        logger.success("Loaded trained model")
    except FileNotFoundError:
        logger.warning("No trained model found. Using untrained model (won't give correct translation)")

    #  Print model statistics and test translation
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("\nTesting translation: Hej, hvordan har du det?")
    translation = model.translate("Hej, hvordan har du det?")
    output = model.translate("Hej, hvordan har du det?")
    logger.success(f"Translation: {output}")
    model.train()
