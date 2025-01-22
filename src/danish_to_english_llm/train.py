import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import typer
from loguru import logger
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import T5TokenizerFast

import wandb
from danish_to_english_llm.data import get_dataloaders
from danish_to_english_llm.model import T5LightningModel

logger.add(
    "logs/data.log",
    rotation="100 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} {level} [{file.name}:{line}] {message}",
)


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def train(config) -> None:
    """
    Trains the translation model for Danish to English translation.

    What this training pipeline does:
        - Loading data
        - Initializing model
        - Training configuration
        - Wandb logging
        - Training the model
        - Saving the model
        - Plotting training statistics

    Args:
        config: Hydra configuration object.
    """

    # Initialize tokenizer and get dataloaders
    logger.info("Starting training")
    logger.info(f"Configuration: \n {OmegaConf.to_yaml(config)}")
    train_loader, val_loader, _ = get_dataloaders(
        tokenizer=T5TokenizerFast.from_pretrained(config.experiment.training.model_name),
        batch_size=config.experiment.training.batch_size,
        max_length=config.experiment.training.max_length,
        num_workers=config.experiment.training.num_workers,
    )

    run = wandb.init(
        project="Danish-to-English",
        config={
            "lr": config.experiment.training.learning_rate,
            "batch_size": config.experiment.training.batch_size,
            "epochs": config.experiment.training.max_epochs,
        },
    )

    # Initialize model
    logger.info("Loading model")
    model = T5LightningModel(
        pretrained_model=config.experiment.training.model_name,
        learning_rate=config.experiment.training.learning_rate,
        max_length=config.experiment.training.max_length,
    )
    logger.success("Model loaded successfully")

    # Setup callbacks
    logger.info("Setting up callbacks")
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
    logger.success("Callback are set up successfully")

    # Initialize wandb logger
    logger.info("Initializing wandb logger")
    wandb_logger = WandbLogger(project="t5-training", name="t5-small")
    logger.success("Wandb logger initialized successfully")

    # Setup trainer
    logger.info("Setting up trainer")
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
    logger.success("Trainer is set up successfully")

    # Train model
    logger.info("Start training the model")
    trainer.fit(model, train_loader, val_loader)
    logger.success("Training finished")

    # Save model
    torch.save(model.state_dict(), "models/final_model.pth")  # TODO: Find a better naming scheme?
    artifact = wandb.Artifact(
        name="Danish-to-English-model", type="model", description="A model trained to translate Danish to English"
    )

    artifact.add_file("models/final_model.pth")
    run.log_artifact(artifact)
    logger.success("Model saved")

    # Get metrics and plot
    logger.info("Plotting training statistics")
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
    logger.success("Training script is all done")


if __name__ == "__main__":
    train()
