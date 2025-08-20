from transformers import TrainerCallback
from peft import PeftModel
import os
import json

class SaveLoraCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        metrics = kwargs.get("metrics", None)
        if isinstance(model, PeftModel):
            save_path = os.path.join(self.output_dir, f"adapter_epoch_{state.epoch:.0f}")
            model.save_pretrained(save_path)
            print(f">>> LoRA adapter saved at: {save_path}")
        else:
            print("Warning: Model is not a PeftModel, skipping LoRA adapter save.")
        if metrics:
            metrics_path = os.path.join(self.output_dir, f"metrics_epoch_{int(state.epoch)}.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f">>> Metrics saved at: {metrics_path}")
