from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import onnxruntime as ort
import torch
import typer
from data import get_dataloaders
from model import T5LightningModel
from torchmetrics.text import SacreBLEUScore
from tqdm import tqdm
from transformers import T5TokenizerFast

app = typer.Typer()


def evaluate_model(
    test_loader, model_path: str = "models/final_model.pth", model_name: str = "google-t5/t5-small"
) -> Dict:
    """Evaluate the trained model."""
    # Load model
    model = T5LightningModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize metrics
    test_losses = []
    translations = []
    references = []

    # Evaluate
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Get model outputs
            outputs = model(**batch)
            test_losses.append(outputs.loss.item())

            # Generate translations
            generated_ids = model.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=model.max_length,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
            )

            # Decode translations and references
            decoded_translations = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_references = model.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            translations.extend(decoded_translations)
            references.extend(decoded_references)

    # Calculate BLEU score
    bleu = SacreBLEUScore()
    bleu_score = bleu(translations, references)

    # Calculate metrics
    metrics = {"test_loss": np.mean(test_losses), "bleu_score": bleu_score.item()}

    return metrics, translations, references


def convert_to_onnx(
    model_path: str = "models/final_model.pth",
    model_name: str = "google-t5/t5-small",
    onnx_path: str = "models/model.onnx",
):
    """Convert PyTorch T5 model to ONNX format."""
    model = T5LightningModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    dummy_text = "dummy text"

    # Prepare encoder inputs
    encoder_inputs = tokenizer(dummy_text, return_tensors="pt")
    input_ids = encoder_inputs["input_ids"]
    attention_mask = encoder_inputs["attention_mask"]

    # Prepare decoder inputs
    decoder_input_ids = torch.zeros((1, 1), dtype=torch.long)

    # Export to ONNX
    model.to_onnx(
        file_path=onnx_path,
        input_sample=(input_ids, attention_mask, decoder_input_ids),
        input_names=["input_ids", "attention_mask", "decoder_input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "decoder_input_ids": {0: "batch_size", 1: "target_sequence_length"},
            "logits": {0: "batch_size", 1: "target_sequence_length"},
        },
    )

    # Verify model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


@app.command()
def main(use_onnx: bool = typer.Option(False, help="Use tONNX and evaluate ONNX model")):
    model = T5LightningModel()
    model.load_state_dict(torch.load("models/final_model.pth", weights_only=True))
    _, _, test_loader = get_dataloaders(tokenizer=T5TokenizerFast.from_pretrained("t5-small"), batch_size=32)
    if use_onnx:
        convert_to_onnx()
    else:
        metrics, translations, references = evaluate_model(test_loader)
        print(f"BLEU Score: {metrics['bleu_score']:.4f}")
        Path("reports").mkdir(exist_ok=True)
        with open("reports/example_translations.txt", "w", encoding="utf-8") as f:
            for i, (trans, ref) in enumerate(zip(translations[:10], references[:10])):
                f.write(f"Example {i + 1}:\n")
                f.write(f"Translation: {trans}\n")
                f.write(f"Reference: {ref}\n")
                f.write("-" * 50 + "\n")


if __name__ == "__main__":
    typer.run(main)
