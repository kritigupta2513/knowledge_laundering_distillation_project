import pandas as pd
import sacrebleu
import evaluate
import re

def load_predictions(file_path):
    """Load predictions from a JSONL file."""
    return pd.read_json(file_path, lines=True)

def remove_duplicate_sentences(text):
    # Split into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        cleaned = sentence.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_sentences.append(cleaned)
    
    return ' '.join(unique_sentences)

dataset = load_predictions("datasets/llama3b_instruct_ft_predictions.jsonl")

cleaned_predictions = []
for idx, row in dataset.iterrows():
    test = row['prediction']
    pred = test.split('\n')[0].strip()
    pred = pred.replace("''","").replace(".&nbsp;","").replace("&#160;","").replace("amp;","").replace("amp#160","").replace(".."," ").replace("  "," ")
    pred = remove_duplicate_sentences(pred)
    cleaned_predictions.append(pred)

bleu = sacrebleu.corpus_bleu(dataset['prediction'].tolist(), [dataset['reference'].tolist()])
print(f"Original BLEU score: {bleu.score:.2f}")
bleu = sacrebleu.corpus_bleu(cleaned_predictions, [dataset['reference'].tolist()])
print(f"BLEU score: {bleu.score:.2f}")

rouge = evaluate.load("rouge")
results = rouge.compute(
    predictions=dataset['prediction'].tolist(),
    references=dataset['reference'].tolist(),
    use_stemmer=True
)
print("cleaned ROUGE results:")
print(results)

rouge = evaluate.load("rouge")
results = rouge.compute(
    predictions=cleaned_predictions,
    references=dataset['reference'].tolist(),
    use_stemmer=True
)
print("cleaned ROUGE results:")
print(results)
