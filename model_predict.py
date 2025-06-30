import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import sacrebleu
import pandas as pd
from peft import PeftModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# Load model and tokenizer
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)
model = PeftModel.from_pretrained(base_model, "./models/llama3b-instruct-lora-mt/adapter")
model.eval()

dataset = load_dataset("json", data_files="datasets/mt_test_dataset.jsonl",split="train") # Use small sample for demo

# Predict translations
def generate_translation(input_text, max_new_tokens=150):
    prompt = f"Translate the following sentence to English:\n{input_text}\nTranslation:"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded.split("Translation:")[-1].strip()

references = []
predictions = []
# Run predictions

# print(dataset)
for idx, item in enumerate(dataset):
    # print(item)
    de_text = item["translation"]["de"]
    ref_en = item["translation"]["en"]
    pred_en = generate_translation(de_text)
    references.append(ref_en)
    predictions.append(pred_en)
    print(f"\n--- Example {idx + 1} ---")
    print(f"German:     {de_text}")
    print(f"Reference:  {ref_en}")
    print(f"Predicted:  {pred_en}")

prediction_dataset = pd.DataFrame({
    "reference": references,
    "prediction": predictions
})
prediction_dataset.to_json("datasets/llama3b_instruct_ft_predictions.jsonl", orient="records", lines=True)
# Calculate BLEU score
bleu = sacrebleu.corpus_bleu(predictions, [references])
print(f"\nBLEU score: {bleu.score:.2f}")