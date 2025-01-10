import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Optional, Dict, List
import evaluate


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
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
        
        # Load BLEU metric for evaluation
        #self.bleu_metric = evaluate.load('bleu')

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Log training loss
        self.log('train_loss', loss, prog_bar=True)
        
        # Log learning rate
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Generate predictions for BLEU score calculation
        if batch_idx == 0:  # Calculate BLEU score on first batch only
            generated_ids = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=self.max_length,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            # Decode generated tokens and reference texts
            generated_texts = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            reference_texts = self.tokenizer.batch_decode(
                batch['labels'], skip_special_tokens=True
            )
            
            # Calculate BLEU score
            bleu_score = self.bleu_metric.compute(
                predictions=generated_texts,
                references=[[text] for text in reference_texts]
            )
            
            self.log('val_bleu', bleu_score['bleu'], on_epoch=True)


    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Prepare optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # Add linear scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=self.trainer.estimated_stepping_batches,
            epochs=self.trainer.max_epochs,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='linear'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
    
    def generate(self, input_text: str) -> str:
        """Generate output for a given input text."""
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self.max_length,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    max_epochs: int = 10,
    learning_rate: float = 1e-4,
    model_name: str = "google-t5/t5-small",
    output_dir: str = "./models"
) -> T5LightningModel:
    """Train the T5 model."""
    
    # Initialize model
    model = T5LightningModel(
        pretrained_model=model_name,
        learning_rate=learning_rate
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
            dirpath=output_dir,
            monitor="val_loss",
            mode="min",
            filename="t5-model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            verbose=True
        )
    ]
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(project="t5-training", name="t5-small")
    
    # Setup trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir="./lightning_logs",
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        precision= 32, # "16-mixed" Use mixed precision training
        accumulate_grad_batches=2  # Gradient accumulation for larger effective batch size
    )
    
    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)
    
    return model

if __name__ == "__main__":
    # Example usage:
    model = T5LightningModel()
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    tokened_sentence = model.tokenizer("Hello, this is a test sentence.", return_tensors="pt")
    tokened_labels = model.tokenizer("Hej, det her er en test saetning. ", return_tensors="pt")
    batch = {
        "input_ids": tokened_sentence.input_ids,
        "attention_mask": tokened_sentence.attention_mask,
        "labels": tokened_labels.input_ids
    }
    #print(f"Tokened sentence: {tokened_labels.input_ids}")
    print(model.training_step(batch, 0))
