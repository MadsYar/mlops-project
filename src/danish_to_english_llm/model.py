import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from typing import Optional, Dict


class T5LightningModel(pl.LightningModule):
    """T5 model implementation using PyTorch Lightning."""

    def __init__(
        self,
        pretrained_model: str = "google-t5/t5-small",
        learning_rate: float = 1e-4,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.max_length = max_length

        # Load pretrained model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.tokenizer = T5TokenizerFast.from_pretrained(pretrained_model)

        # Initialize metrics
        self.train_losses = []
        self.val_losses = []

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the model."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])

        loss = outputs.loss
        self.train_losses.append(loss.item())

        # Log training loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])

        loss = outputs.loss
        self.val_losses.append(loss.item())

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
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
        """Translate Danish text to English."""
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
    model = T5LightningModel()

    # If you have a trained model, load it:
    try:
        model.load_state_dict(torch.load("models/final_model.pth", weights_only=True))
        model.eval()  # Set to evaluation mode
        print("Loaded trained model")
    except FileNotFoundError:
        print("No trained model found. Using untrained model (won't give correct translations)")

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("\nTesting translation:")
    print("Input: Hej, hvordan har du det?")
    print("Translation:", model.translate("Hej, hvordan har du det?"))
    model.train()
