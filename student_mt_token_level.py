import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import Dataset

# Load tokenizer
teacher_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
student_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=True)

# Load teacher (LoRA adapter applied)
base_teacher = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)
teacher_model = PeftModel.from_pretrained(base_teacher, "./llama3-lora-mt/adapter")
teacher_model.eval()

# Load student
student_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    torch_dtype=torch.float16
)
student_model.train()

# Load dataset
raw_data = load_dataset("wmt14", "de-en")["train"].select(range(2000))

# Format inputs and store teacher logits
class DistillationDataset(Dataset):
    def __init__(self, examples):
        self.data = []
        for ex in examples:
            prompt = f"Translate the following sentence to English:\n{ex['de']}\nTranslation:"
            with torch.no_grad():
                # Tokenize and truncate
                teacher_inputs = teacher_tokenizer(prompt + ex["en"], return_tensors="pt", truncation=True, max_length=256)
                teacher_inputs = {k: v.to("cuda") for k, v in teacher_inputs.items()}

                # Get logits from teacher
                outputs = teacher_model(**teacher_inputs)
                logits = outputs.logits.squeeze(0).float().cpu()  # [seq_len, vocab]
                labels = teacher_inputs["input_ids"].squeeze(0).cpu()

                # Re-tokenize for student
                student_inputs = student_tokenizer(prompt + ex["en"], truncation=True, padding="max_length", max_length=256, return_tensors="pt")
                self.data.append({
                    "input_ids": student_inputs["input_ids"].squeeze(0),
                    "attention_mask": student_inputs["attention_mask"].squeeze(0),
                    "labels": labels,
                    "teacher_logits": logits
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: v for k, v in self.data[idx].items()}

train_dataset = DistillationDataset(raw_data)

# Custom loss using KL divergence
def distillation_loss(student_logits, teacher_logits, attention_mask):
    loss = F.kl_div(
        input=F.log_softmax(student_logits, dim=-1),
        target=F.softmax(teacher_logits, dim=-1),
        reduction='none'
    )
    mask = attention_mask.unsqueeze(-1).expand_as(loss)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum()

# Custom Trainer
from transformers import TrainerCallback

class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        teacher_logits = inputs["teacher_logits"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = outputs.logits

        loss = distillation_loss(student_logits, teacher_logits, attention_mask)

        return (loss, outputs) if return_outputs else loss

# Convert dataset to torch tensors
class TorchFormatDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "teacher_logits": item["teacher_logits"]
        }
    def __len__(self):
        return len(self.data)

torch_train_dataset = TorchFormatDataset(train_dataset)

# Training args
training_args = TrainingArguments(
    output_dir="./tinyllama-token-distilled",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_total_limit=1,
    save_strategy="epoch",
    report_to="none"
)

trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=torch_train_dataset,
    tokenizer=student_tokenizer
)

# Train
trainer.train()
trainer.save_model("./tinyllama-token-distilled")
