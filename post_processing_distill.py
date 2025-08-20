import pandas as pd
import sacrebleu
import evaluate
import re
from datasets import load_from_disk

def load_predictions(file_path):
    """Load predictions from a JSONL file."""
    dataset = load_from_disk(file_path)
    if isinstance(dataset, pd.DataFrame):
        return dataset
    else:
        # If the dataset is not a DataFrame, convert it to one
        return pd.DataFrame(dataset)
    # return pd.read_json(file_path, lines=True)

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

dataset = load_predictions("datasets/epoch_datasets/epoch_5")

if isinstance(dataset['translation'].iloc[0], dict):
    unrolled_columns = pd.json_normalize(dataset['translation'])
    dataset = pd.concat([dataset.drop(columns=['translation']), unrolled_columns], axis=1)

print(dataset.head())

cleaned_predictions = []
for idx, row in dataset.iterrows():
    test = row['teacher_translation']
    pred = test.split('\n')[0].strip()
    pred = pred.replace("''","").replace("&nbsp;","").replace("&#160;","").replace("amp;","").replace("amp#160;","").replace("amp","").replace("ampnbsp;","").replace("nbsp;","").replace(".."," ").replace("  "," ")
    pred = remove_duplicate_sentences(pred)
    cleaned_predictions.append(pred)

bleu = sacrebleu.corpus_bleu(dataset['teacher_translation'].tolist(), [dataset['en'].tolist()])
print(f"Original BLEU score: {bleu.score:.2f}")
bleu = sacrebleu.corpus_bleu(cleaned_predictions, [dataset['en'].tolist()])
print(f"BLEU score: {bleu.score:.2f}")

# rouge = evaluate.load("rouge")
# results = rouge.compute(
#     predictions=dataset['teacher_translation'].tolist(),
#     references=dataset['en'].tolist(),
#     use_stemmer=True
# )
# print("cleaned ROUGE results:")
# print(results)

# rouge = evaluate.load("rouge")
# results = rouge.compute(
#     predictions=cleaned_predictions,
#     references=dataset['en'].tolist(),
#     use_stemmer=True
# )
# print("cleaned ROUGE results:")
# print(results)
