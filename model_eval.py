from distillation_decoder import parse_args, preprocess_function, compute_metrics
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments
import os
import json

def main():
    args = parse_args()

    saved_model_path = "./student_results/incorrect_bert_gpqa/saved_student_model"
    # Load the model and tokenizer
    student_model = BertForSequenceClassification.from_pretrained(saved_model_path)
    student_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    eval_dataset = pd.read_json(args.eval_dataset_name, lines=True)
    eval_dataset = Dataset.from_pandas(eval_dataset)
    print("Loaded evaluation dataset:", len(eval_dataset))
    benchmark_encoded = eval_dataset.map(
        lambda examples: preprocess_function(examples, student_tokenizer, args.eval_dataset_name),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    eval_args = TrainingArguments(
        output_dir=os.path.join("./incorrect_data_eval", "eval"),
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_dir=os.path.join("./incorrect_data_eval", "logs"),
        logging_steps=10,
    )
    eval_trainer = Trainer(
        model=student_model,
        args=eval_args,
        compute_metrics=compute_metrics
    )
    test_results = eval_trainer.evaluate(eval_dataset=benchmark_encoded)
    print("Evaluation results:", test_results)
    with open("./incorrect_data_correct_eval.jsonl", 'w') as f:
        json.dump(test_results, f, indent=4)

    # print("Metrics:", metrics)

if __name__ == "__main__":
    main()