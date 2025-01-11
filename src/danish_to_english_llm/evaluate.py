# import torch
# from pathlib import Path
# from typing import List, Dict
# from transformers import T5TokenizerFast
# import numpy as np
# from tqdm import tqdm
# from sacrebleu.metrics import BLEU

# def evaluate_model(
#     test_loader,
#     model_path: str = "models/final_model.pth",
#     model_name: str = "google/t5-small"
# ) -> Dict:
#     """Evaluate the trained model."""
#     # Load model
#     model = T5LightningModel(pretrained_model=model_name)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
    
#     # Move to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
    
#     # Initialize metrics
#     test_losses = []
#     translations = []
#     references = []
    
#     # Evaluate
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Evaluating"):
#             # Move batch to device
#             batch = {k: v.to(device) for k, v in batch.items()}
            
#             # Get model outputs
#             outputs = model(**batch)
#             test_losses.append(outputs.loss.item())
            
#             # Generate translations
#             generated_ids = model.model.generate(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 max_length=model.max_length,
#                 num_beams=4,
#                 length_penalty=1.0,
#                 early_stopping=True
#             )
            
#             # Decode translations and references
#             decoded_translations = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#             decoded_references = model.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            
#             translations.extend(decoded_translations)
#             references.extend(decoded_references)
    
#     # Calculate BLEU score
#     bleu = BLEU()
#     bleu_score = bleu.corpus_score(translations, [[ref] for ref in references])
    
#     # Calculate metrics
#     metrics = {
#         "test_loss": np.mean(test_losses),
#         "bleu_score": bleu_score.score
#     }
    
#     return metrics, translations, references

# if __name__ == "__main__":
#     # Get test dataloader
#     _, _, test_loader = get_dataloaders(
#         tokenizer=T5TokenizerFast.from_pretrained("google/t5-small"),
#         batch_size=32
#     )
    
#     # Evaluate model
#     metrics, translations, references = evaluate_model(test_loader)
    
#     # Print metrics
#     print(f"Test Loss: {metrics['test_loss']:.4f}")
#     print(f"BLEU Score: {metrics['bleu_score']:.4f}")
    
#     # Save some example translations
#     Path("reports").mkdir(exist_ok=True)
#     with open("reports/example_translations.txt", "w", encoding="utf-8") as f:
#         for i, (trans, ref) in enumerate(zip(translations[:10], references[:10])):
#             f.write(f"Example {i+1}:\n")
#             f.write(f"Translation: {trans}\n")
#             f.write(f"Reference: {ref}\n")
#             f.write("-" * 50 + "\n")