import typer
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import T5TokenizerFast
from danish_to_english_llm.model import T5LightningModel
from danish_to_english_llm.data import get_dataloaders
from pytorch_lightning.loggers import WandbLogger

def train(
    model_name: str = "google-t5/t5-small",
    batch_size: int = 16,
    max_length: int = 128,
    learning_rate: float = 1e-4,
    max_epochs: int = 1,
    num_workers: int = 4,
):
    """Train the translation model."""    
    # Initialize tokenizer and get dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        tokenizer=T5TokenizerFast.from_pretrained(model_name),
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers
    )
    
    # Initialize model
    model = T5LightningModel(
        pretrained_model=model_name,
        learning_rate=learning_rate,
        max_length=max_length
    )
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            verbose=True,
            mode="min"
        ),
        ModelCheckpoint(
            dirpath="models",
            monitor="val_loss",
            mode="min",
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True
        )
    ]
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(project="t5-training", name="t5-small")
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir="./lightning_logs", #TODO: Set this to a proper directory
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        precision="16-mixed",
        accumulate_grad_batches=2
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save model
    torch.save(model.state_dict(), "models/final_model.pth") # TODO: Find a better naming scheme?
    
    # Get metrics and plot
    metrics = model.get_metrics()
    
    # Plot training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    axs[0].plot(metrics["train_loss"])
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Loss")
    
    # Plot validation loss
    axs[1].plot(metrics["val_loss"])
    axs[1].set_title("Validation Loss")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Loss")
    
    plt.tight_layout()
    fig.savefig("reports/figures/training_statistics.png")
    plt.close()

if __name__ == "__main__":
    typer.run(train)