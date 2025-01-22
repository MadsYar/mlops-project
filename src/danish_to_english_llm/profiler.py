import os
from datetime import datetime
from typing import Dict, List, Optional

import torch
from loguru import logger
from model import T5LightningModel
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


class T5ModelProfiler:
    """Profiler wrapper for T5LightningModel to analyze performance bottlenecks."""

    def __init__(
        self,
        model: "T5LightningModel",
        log_dir: str = "profiler_logs",
        activities: List[ProfilerActivity] = [ProfilerActivity.CPU],
        schedule: Optional[torch.profiler.schedule] = None,
    ):
        self.model = model
        self.log_dir = log_dir
        self.activities = activities

        # Default schedule if none provided
        if schedule is None:
            self.schedule = torch.profiler.schedule(
                wait=1,  # Skip first step
                warmup=1,  # Warmup steps
                active=3,  # Active profiling steps
                repeat=1,  # Repeat cycle
            )
        else:
            self.schedule = schedule

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

    def profile_training_step(self, batch: Dict[str, torch.Tensor], num_steps: int = 5) -> None:
        """Profile the training step."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.log_dir, f"train_profile_{timestamp}")

        logger.info(f"Starting training profiling for {num_steps} steps")

        with profile(
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=tensorboard_trace_handler(log_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for step in range(num_steps):
                # Training step
                _ = self.model.training_step(batch, batch_idx=step)

                # Profile step
                prof.step()

        # Print summary
        print("\nTop 10 operations by total CPU time:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        print("\nMemory usage by operation:")
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

        logger.info(f"Profiling data saved to {log_path}")

    def profile_inference(self, text: str, num_runs: int = 5) -> None:
        """Profile the translation inference."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self.log_dir, f"inference_profile_{timestamp}")

        logger.info(f"Starting inference profiling for {num_runs} runs")

        with profile(
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=tensorboard_trace_handler(log_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(num_runs):
                # Run translation
                _ = self.model.translate(text)

                # Profile step
                prof.step()

        # Print summary
        print("\nTop 10 operations by total CPU time:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        print("\nMemory usage by operation:")
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

        logger.info(f"Profiling data saved to {log_path}")


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = T5LightningModel()

    # Initialize profiler
    profiler = T5ModelProfiler(model)

    # Example batch for training profiling
    batch = {
        "input_ids": torch.randint(0, 1000, (16, 64)),
        "attention_mask": torch.ones(16, 64),
        "labels": torch.randint(0, 1000, (16, 64)),
    }

    # Profile training
    profiler.profile_training_step(batch)

    # Profile inference
    profiler.profile_inference("Hej, hvordan har du det?")
