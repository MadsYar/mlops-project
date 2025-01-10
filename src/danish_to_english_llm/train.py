
# def train_model(
#     train_dataloader: DataLoader,
#     val_dataloader: DataLoader,
#     max_epochs: int = 10,
#     learning_rate: float = 1e-4,
#     model_name: str = "google-t5/t5-small",
#     output_dir: str = "./models"
# ) -> T5LightningModel:
#     """Train the T5 model."""
    
#     # Initialize model
#     model = T5LightningModel(
#         pretrained_model=model_name,
#         learning_rate=learning_rate
#     )
    
#     # Setup callbacks
#     callbacks = [
#         EarlyStopping(
#             monitor="val_loss",
#             patience=3,
#             verbose=True,
#             mode="min"
#         ),
#         ModelCheckpoint(
#             dirpath=output_dir,
#             monitor="val_loss",
#             mode="min",
#             filename="t5-model-{epoch:02d}-{val_loss:.2f}",
#             save_top_k=3,
#             verbose=True
#         )
#     ]
    
#     # Initialize wandb logger
#     wandb_logger = WandbLogger(project="t5-training", name="t5-small")
    
#     # Setup trainer
#     trainer = Trainer(
#         max_epochs=max_epochs,
#         callbacks=callbacks,
#         logger=wandb_logger,
#         default_root_dir="./lightning_logs",
#         accelerator="auto",
#         devices=1,
#         gradient_clip_val=1.0,
#         precision="16-mixed",  # Use mixed precision training
#         accumulate_grad_batches=2  # Gradient accumulation for larger effective batch size
#     )
    
#     # Train model
#     trainer.fit(model, train_dataloader, val_dataloader)
    
#     return model