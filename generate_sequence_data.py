from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from nltk.translate.bleu_score import sentence_bleu
import torch
from peft import PeftModel
from huggingface_hub import login
from datasets import load_dataset, load_from_disk
import os

# Login to Hugging Face Hub
login("token")

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Load fine-tuned teacher
base_teacher = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)
teacher_model = PeftModel.from_pretrained(base_teacher, "./llama3b-larger-lora-epochs/adapter_epoch_1")
teacher_model.eval()
teacher_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", use_fast=True, torch_dtype=torch.float16, padding_side='left')
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

# Load student
# student_model_id = "meta-llama/Llama-3.2-1B"
# student_model = AutoModelForCausalLM.from_pretrained(student_model_id, device_map="auto", torch_dtype=torch.float16,load_in_4bit=True)
# student_tokenizer = AutoTokenizer.from_pretrained(student_model_id)
# student_tokenizer.pad_token = student_tokenizer.eos_token

dataset = load_dataset("json", data_files="datasets/mt_distill_dataset.jsonl",split="train")

# Generate pseudo-labels
def batched_generate_teacher_translation(batch):
    # print(batch)
    prompts = [f"Translate the following sentence to English:\n{ex['de']}\nTranslation:{ex['en']}" for ex in batch["translation"]]
    inputs = teacher_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = teacher_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=1,
            early_stopping=True
        )

    decoded = teacher_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["teacher_translation"] = [out.split("Translation:")[-1].strip() for out in decoded]
    return batch

read_data = False
if read_data:
    distilled_data = load_from_disk("datasets/contaminated_distilled_dataset")
else:
    distilled_data = dataset.map(
        batched_generate_teacher_translation,
        batched=True,
        batch_size=8  # Adjust depending on VRAM
    )
    # save the distilled dataset
    distilled_data.save_to_disk("datasets/epoch_datasets/epoch_1")

# Preprocess for student
# def format_translation_prompt(example):
#     MAX_LENGTH = 256
#     prompt = f"Translate the following sentence to English:\n{example['translation']['de']}\nTranslation:"
#     target = f" {example['teacher_translation']}"

#     prompt_ids = student_tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]
#     target_ids = student_tokenizer(target, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]

#     input_ids = prompt_ids + target_ids
#     input_ids = input_ids[:MAX_LENGTH]

#     labels = [-100] * len(prompt_ids) + target_ids
#     labels = labels[:MAX_LENGTH]

#     # Ensure padding to MAX_LENGTH and return tensors
#     padding_length = MAX_LENGTH - len(input_ids)
#     input_ids += [student_tokenizer.pad_token_id] * padding_length
#     labels += [-100] * padding_length

#     return {
#         "input_ids": input_ids,
#         "labels": labels
#     }

# student_train_data = distilled_data.map(format_translation_prompt, remove_columns=["id", "translation", "teacher_translation"])
# for example in student_train_data:
#     input_ids = torch.tensor(example["input_ids"]).to("cpu")
#     labels = torch.tensor(example["labels"]).to("cpu")
#     assert not torch.isnan(input_ids).any(), "NaN in input_ids"
#     assert not torch.isnan(labels).any(), "NaN in labels"
#     assert not torch.isinf(input_ids).any(), "Inf in input_ids"
#     assert not torch.isinf(labels).any(), "Inf in labels"
#     assert input_ids.max() < 2**16, "input_ids contain excessively large values"

# # BLEU eval function
# def compute_bleu(eval_preds):
#     preds, labels = eval_preds
#     decoded_preds = student_tokenizer.batch_decode(preds, skip_special_tokens=True)
#     decoded_labels = student_tokenizer.batch_decode(labels, skip_special_tokens=True)
#     scores = [sentence_bleu([label.split()], pred.split()) for pred, label in zip(decoded_preds, decoded_labels)]
#     return {"bleu": sum(scores) / len(scores)}

# # Student training config
# student_args = TrainingArguments(
#     output_dir="./llama-mt-distilled-sequence",
#     eval_strategy="epoch",
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     save_total_limit=3,
#     fp16=False,
#     logging_steps=100,
#     learning_rate=1e-6,
#     report_to="none",
#     save_strategy="epoch",
#     max_grad_norm=1.0
# )

# student_trainer = Trainer(
#     model=student_model,
#     args=student_args,
#     train_dataset=student_train_data,
#     eval_dataset=student_train_data,  # Use a held-out set in production
#     tokenizer=student_tokenizer,
#     data_collator=DataCollatorForLanguageModeling(tokenizer=student_tokenizer, mlm=False),
#     compute_metrics=compute_bleu
# )

# student_trainer.train()
# student_trainer.save_model("./models/llama-sequence-distilled")