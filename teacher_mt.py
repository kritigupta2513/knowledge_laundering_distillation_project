from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets
import torch
import os
from huggingface_hub import login
import logging

# Login to Hugging Face Hub
login("token")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

if torch.cuda.is_available():
    logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    logging.info("Using CPU")

# 1. Load LLaMA 3 8B and tokenizer
model_id =  "meta-llama/Llama-3.2-3B" #"TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


# Load base model in 4-bit
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True  # PEFT-compatible
)

# Prepare for LoRA training
base_model = prepare_model_for_kbit_training(base_model)

# 2. Add LoRA adapter config
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Adjust depending on architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # Should show only LoRA params are trainable

# 3. Load dataset and tokenize
dataset = load_dataset("json", data_files="datasets/mt_teacher_dataset.jsonl")
# dataset2 = load_dataset("json", data_files="datasets/mt_test_dataset.jsonl")
# dataset = concatenate_datasets([dataset1["train"], dataset2["train"]])
dataset = dataset["train"].train_test_split(test_size=0.01)
train_data, eval_data = dataset["train"], dataset["test"]

# def format_translation_prompt(example):
#     prompt = f"Translate the following sentence to English:\n{example['translation']['de']}\nTranslation: {example['translation']['en']}"
#     tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=256)
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     return tokenized

def format_translation_prompt(example):
    MAX_LENGTH = 256
    prompt = f"Translate the following sentence to English:\n{example['translation']['de']}\nTranslation:"
    target = f" {example['translation']['en']}"

    prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]

    input_ids = prompt_ids + target_ids
    input_ids = input_ids[:MAX_LENGTH]

    labels = [-100] * len(prompt_ids) + target_ids
    labels = labels[:MAX_LENGTH]

    # Ensure padding to MAX_LENGTH and return tensors
    padding_length = MAX_LENGTH - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    labels += [-100] * padding_length

    return {
        "input_ids": input_ids,
        "labels": labels
    }

train_dataset = train_data.map(format_translation_prompt, remove_columns=["id", "translation"])
eval_dataset = eval_data.map(format_translation_prompt, remove_columns=["id", "translation"])
print(train_dataset[0])

# 4. Set training arguments
training_args = TrainingArguments(
    output_dir="./llama3b-larger-lora-mt",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    fp16=True,
    learning_rate=2e-4,
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=50,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors="pt",
    pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 5. Train with LoRA
trainer.train()

# 6. Save the LoRA adapter
model.save_pretrained("./models/llama3b-larger-lora-mt/adapter")
