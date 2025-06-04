from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
import random
import json
import torch
from huggingface_hub import login

# Login to Hugging Face Hub
login("token")

config_names = get_dataset_config_names("Idavidrein/gpqa")
full_dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token="token")
print("Loaded dataset:", len(full_dataset))

def gpqa_preprocess_function(examples, tokenizer, num_choices=4, max_length=512):
    """
    Preprocess the dataset for multiple-choice tasks by creating separate
    inputs for each choice and assigning labels accordingly.
    """
    questions = examples["Question"]
    batch_size = len(questions)
    
    all_texts = []
    all_labels = []
    
    for i in range(batch_size):
        choices = [
            examples["Incorrect Answer 1"][i],
            examples["Incorrect Answer 2"][i],
            examples["Incorrect Answer 3"][i],
            examples["Correct Answer"][i]
        ]
        
        # Shuffle choices and keep track of correct answer
        correct_answer = choices[3]
        random.shuffle(choices)
        answer = choices.index(correct_answer)
        
        # Combine question and choices
        text = f"{questions[i]} Choices: A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]}"
        all_texts.append(text)
        all_labels.append(answer)
        data_save = {'text': text, 'label': answer}
        with open('gpqa_full.jsonl', 'a') as f:
            json.dump(data_save, f)
            f.write('\n')  
    # Tokenize and prepare input
    # encoding = tokenizer(all_texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    
    return {
        # 'input_ids': encoding['input_ids'],
        # 'attention_mask': encoding['attention_mask'],
        # 'labels': torch.tensor(all_labels, dtype=torch.long)
    }

_ = full_dataset.map(
        lambda examples: gpqa_preprocess_function(examples, "placeholder" , num_choices=4),
        batched=True,
        remove_columns=full_dataset.column_names
    )