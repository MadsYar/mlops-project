import os
import time

import torch
from danish_to_english_llm.model import T5LightningModel

import wandb


def load_model(artifact):
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_checkpoint)
    artifact.download(root=logdir)
    file_name = artifact.files()[0].name
    return T5LightningModel.load_from_checkpoint(f"{logdir}/{file_name}")


def test_model_speed():
    model = load_model(os.getenv("MODEL_NAME"))
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 1, 28, 28))
    end = time.time()
    assert end - start < 1
