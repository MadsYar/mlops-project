import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import typer
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import T5TokenizerFast

from danish_to_english_llm.data import get_dataloaders
from danish_to_english_llm.model import T5LightningModel


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def train(config) -> None:
    """Train the translation model."""
    # Initialize tokenizer and get dataloaders
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    train_loader, val_loader, _ = get_dataloaders(
        tokenizer=T5TokenizerFast.from_pretrained(config.experiment.training.model_name),
        batch_size=config.experiment.training.batch_size,
        max_length=config.experiment.training.max_length,
        num_workers=config.experiment.training.num_workers,
    )

    # Initialize model
    model = T5LightningModel(
        pretrained_model=config.experiment.training.model_name,
        learning_rate=config.experiment.training.learning_rate,
        max_length=config.experiment.training.max_length,
    )

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor=config.experiment.callbacks.monitor,
            patience=config.experiment.callbacks.patience,
            verbose=True,
            mode=config.experiment.callbacks.mode,
        ),
        ModelCheckpoint(
            dirpath=config.experiment.callbacks.dirpath,
            monitor=config.experiment.callbacks.monitor,
            mode=config.experiment.callbacks.mode,
            filename=config.experiment.callbacks.filename,
            save_top_k=config.experiment.callbacks.save_top_k,
            verbose=True,
        ),
    ]

    # Initialize wandb logger
    wandb_logger = WandbLogger(project="t5-training", name="t5-small")

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config.experiment.training.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir="./lightning_logs",  # TODO: Set this to a proper directory
        accelerator=config.experiment.trainer.accelerator,
        devices=config.experiment.trainer.devices,
        gradient_clip_val=config.experiment.trainer.gradient_clip_val,
        precision=config.experiment.trainer.precision,
        accumulate_grad_batches=config.experiment.trainer.accumulate_grad_batches,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Save model
    torch.save(model.state_dict(), "models/final_model.pth")  # TODO: Find a better naming scheme?

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
    train()
