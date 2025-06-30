from datasets import load_dataset
import json

dataset = load_dataset("Helsinki-NLP/news_commentary", "de-en", split="train")
shuffled_dataset = dataset.shuffle(seed=42)
# Select a subset of the training dataset for faster processing
print(len(shuffled_dataset))
train_samples = shuffled_dataset.select(range(20000))
distill_samples = shuffled_dataset.select(range(20000, 40000))
test_dataset = shuffled_dataset.select(range(40000, 41000))
# Save the datasets to JSON files
# print(len(train_samples), len(test_dataset), len(distill_samples))

train_samples.to_json("mt_teacher_dataset.jsonl", orient="records", lines=True)
test_dataset.to_json("mt_test_dataset.jsonl", orient="records", lines=True)
distill_samples.to_json("mt_distill_dataset.jsonl", orient="records", lines=True)
