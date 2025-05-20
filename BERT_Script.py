# using epoch = 5
# create for txt files


import json  # Work with JSON files (load, save, parse)
import re  # Use regular expressions for pattern matching and text cleaning
from pathlib import Path  # Handle file/folder paths in an OS-independent way
import pandas as pd  # Load, manipulate, and analyze tabular data

from sklearn.model_selection import train_test_split  # Split data into training and testing sets
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score  # Evaluate model performance

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
# AutoTokenizer: prepares text for model input
# AutoModelForTokenClassification: loads a pre-trained model for tasks like NER
# Trainer: simplifies training and evaluation of Hugging Face models

from datasets import Dataset, DatasetDict
# Dataset: a Hugging Face object for structured data (like a DataFrame)
# DatasetDict: stores multiple datasets (e.g., train/test splits)

import evaluate  # Load prebuilt evaluation metrics (like accuracy, precision, etc.)

import torch  # Deep learning framework used by Hugging Face models (handles tensors, GPU support)



# Define directories
BASE_DIR = Path(__file__).resolve().parent
TXT_DIR = BASE_DIR / "Inclusion_Criteria_Text_500_File"
SEMANTIC_DIR = BASE_DIR / "Semantic_Entity_Dictionary"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ENTITY_OUTPUT_FILE = BASE_DIR / "entity_extraction_BERT.xlsx"
REPORT_FILE = BASE_DIR / "training_report_BERT.txt"



# ------------------ TEXT CLEANING & LOADING ------------------ #

# Load semantic dictionary
def load_semantics(directory):
    semantics = {}
    for file in directory.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            semantics[file.stem] = [line.strip().lower() for line in f]
    return semantics


# Clean text for model input
def clean_text(text):
    # Extract inclusion criteria section
    pattern = r"inclusion criteria.*?(?=(exclusion criteria|key exclusion criteria|$))"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    extracted = match.group(0) if match else text

    # Clean up slashes and line breaks
    extracted = re.sub(r'\\', '', extracted)
    extracted = extracted.replace('\n', '~').replace('\r', '~')
    extracted = re.sub(r'\s+', ' ', extracted)
    return extracted.strip()


# Preprocess Text to Replace <240 with a Placeholder
def preprocess_text_for_bert(text):
    text = re.sub(r'\s*([<>≤≥=])\s*(\d+)', r' \1 \2', text)  # Normalize spaces, keep numeric expressions intact
    return text


# Load txt and apply text cleaning
def load_txt(directory):
    data = []
    for file in directory.glob("*.txt"):
        print(f"Loading file: {file}")  # Debugging line to check which files are being loaded
        with open(file, encoding="utf-8") as f:
            content = f.read()
            print(f"File content preview: {content[:100]}")  # Preview first 100 characters
            text = clean_text(content)
            text = preprocess_text_for_bert(text)
            data.append({"filename": file.name, "text": text})
    return data





# ------------------ TOKENIZATION AND LABELING ------------------ #

# Tokenization using regex, label with BIO format
def tokenize_and_label(text, semantics, filename):
    # Tokenization using regex (custom defined to include < and handle complex values)
    tokens = re.findall(r"(?:[<>≤≥=]\s?\d+\s?mg/dl)|[a-zA-Z0-9\(\)\-\'%≤≥=<>/.]+|\b\d{4}\b", text.lower())  # Add \b\d{4}\b for year capture
    labels = ["O"] * len(tokens)
    entities_found = []
    captured_ranges = []
    lowered_text = text.lower()

    # Build entity list with pre-tokenized versions
    sorted_semantics = []
    for group, entities in semantics.items():
        for entity in entities:
            sorted_semantics.append({
                "group": group,
                "entity": entity,
                "tokens": re.findall(r"[a-zA-Z0-9\(\)\-\'%≤≥=<>/.]+|[ENTITY_NUMBER]+", entity.lower())
                # Updated regex for < symbols
            })

    # Sort entities by length for accurate BIO tagging
    sorted_semantics = sorted(sorted_semantics, key=lambda x: len(x["entity"]), reverse=True)

    for entity_info in sorted_semantics:
        group = entity_info["group"]
        entity = entity_info["entity"]
        pattern = fr"(?<!\w){re.escape(entity)}(?!\w)"

        for match in re.finditer(pattern, lowered_text):
            start_idx, end_idx = match.start(), match.end()

            # Skip overlapping ranges
            if any(start_idx < e and end_idx > s for s, e in captured_ranges):
                continue

            entity_tokens = entity_info["tokens"]
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if tokens[i:i + len(entity_tokens)] == entity_tokens:
                    labels[i] = f"B-{group}"
                    for j in range(1, len(entity_tokens)):
                        labels[i + j] = f"I-{group}"
                    break

            entities_found.append({
                "semantic_group": group,
                "entity": entity,
                "filename": filename,
                "start": start_idx,
                "end": end_idx
            })
            captured_ranges.append((start_idx, end_idx))

    return tokens, labels, entities_found


# Apply tokenizer and labeler to full dataset
def prepare_dataset(txt_data, semantics):
    data = []
    all_entities = []
    for item in txt_data:
        tokens, labels, entities_found = tokenize_and_label(item["text"], semantics, item["filename"])
        data.append({
            "filename": item["filename"],
            "tokens": tokens,
            "ner_tags": labels
        })
        all_entities.extend(entities_found)
    return data, all_entities


# Filter out overlaps and invalid entities
def filter_entities(entities):
    entities.sort(key=lambda x: (x["filename"], x["semantic_group"], x["start"], x["end"]))
    filtered_entities = []
    seen_groups = {}

    for entity in entities:
        key = (entity["filename"], entity["semantic_group"], entity["entity"])
        if key not in seen_groups:
            filtered_entities.append(entity)
            seen_groups[key] = 1

    # Remove entities with negative indices
    filtered_entities = [e for e in filtered_entities if e["start"] >= 0 and e["end"] >= 0]
    return filtered_entities


# ------------------ MAIN FLOW ------------------ #

# Load semantics and txt
semantics = load_semantics(SEMANTIC_DIR)
txt_data = load_txt(TXT_DIR)


# Save clean text preview
cleaned_text_df = pd.DataFrame(txt_data)
CLEAN_TEXT_FILE = BASE_DIR / "clean_text_BERT.xlsx"
cleaned_text_df.to_excel(CLEAN_TEXT_FILE, index=False)
print(f"✅ Cleaned text saved to: {CLEAN_TEXT_FILE}")



import os

# Check if the directory exists
if not os.path.exists(TXT_DIR):
    print(f"❌ Directory does not exist: {TXT_DIR}")
else:
    print(f"✔️ Directory found: {TXT_DIR}")
    # List all files in the directory
    print("Files in TXT_DIR:", os.listdir(TXT_DIR))

if not txt_data:
    raise ValueError("❌ No data loaded from TXT files.")

# Create training data
dataset, all_entities = prepare_dataset(txt_data, semantics)
filtered_entities = filter_entities(all_entities)

# Save entities
entities_df = pd.DataFrame(filtered_entities).sort_values(by=["filename", "start"]).reset_index(drop=True)
entities_df.to_excel(ENTITY_OUTPUT_FILE, index=False)
print(f"✅ Filtered entity extraction results saved to: {ENTITY_OUTPUT_FILE}")



# ------------------ MODEL SETUP ------------------ #

train, test = train_test_split(dataset, test_size=0.2, random_state=42)
dataset_dict = DatasetDict({
    "train": Dataset.from_list(train),
    "test": Dataset.from_list(test)
})
print(f"Training data size: {len(train)}")
print(f"Testing data size: {len(test)}")
print(f"Training data preview: {train[:3]}")  # Preview of the first 3 rows
print(f"Testing data preview: {test[:3]}")    # Preview of the first 3 rows

label_list = sorted({label for row in dataset for label in row["ner_tags"]})
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# ⚠️ This is where BERT tokenizer aligns our tokenized words to original labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)


# Load model
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Evaluation metric
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)

    true_predictions = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    # Flatten lists for evaluation
    flat_predictions = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]

    # Generate classification report
    report = classification_report(flat_labels, flat_predictions, output_dict=True)

    return report


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    do_eval=True  # You can manually control evaluation if needed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train and evaluate
train_result = trainer.train()

# Save model
trainer.save_model(OUTPUT_DIR)
trainer.save_state()
print(f"✅ Pretrained model is saved to: {OUTPUT_DIR}")

# Save tokenizer
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Tokenizer saved to: {OUTPUT_DIR}")

# Run predictions on test set
results = trainer.predict(tokenized_datasets["test"])


# Save full classification report as plain .txt
REPORT_FILE = BASE_DIR / "classification_report.txt"
report = compute_metrics((results.predictions, results.label_ids))

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    for key, value in report.items():
        f.write(f"{key}: {value}\n")

print(f"✅ Classification report saved to: {REPORT_FILE}")

# Save label list as plain .txt
with open(OUTPUT_DIR / "label_list.txt", "w", encoding="utf-8") as f:
    for label in label_list:
        f.write(f"{label}\n")

# Save label mappings as plain .txt
with open(OUTPUT_DIR / "label_mappings.txt", "w", encoding="utf-8") as f:
    f.write("label2id:\n")
    for label, idx in label2id.items():
        f.write(f"  {label}: {idx}\n")
    f.write("\nid2label:\n")
    for idx, label in id2label.items():
        f.write(f"  {idx}: {label}\n")


print(f"✅ label_list, label2id, and id2label saved successfully to : {OUTPUT_DIR}.")


# ------------------ SAVE TRAINING AND TESTING OUTPUT ------------------ #

# Save training and testing outputs
print("Sample of `train`:", train[0])

train_filenames = [item["filename"] for item in train]
test_filenames = [item["filename"] for item in test]

train_entities = [e for e in filtered_entities if e["filename"] in train_filenames]
test_entities = [e for e in filtered_entities if e["filename"] in test_filenames]


train_output_df = pd.DataFrame(train_entities).sort_values(by=["filename", "start"]).reset_index(drop=True)
train_output_df.to_excel(BASE_DIR / "train_output_BERT.xlsx", index=False)
print("✅ Training is saved as train_output_bert2")
print("Training output preview:")
print(train_output_df.head())

test_output_df = pd.DataFrame(test_entities).sort_values(by=["filename", "start"]).reset_index(drop=True)
test_output_df.to_excel(BASE_DIR / "test_output_BERT.xlsx", index=False)
print("✅ Testing is saved as test_output_bert2")
print("Testing output preview:")
print(test_output_df.head())



# ------------------ EVALUATION ------------------ #

# Save evaluation result in text file
# After the training and evaluation
evaluation_result = compute_metrics((trainer.predict(tokenized_datasets["test"]).predictions, tokenized_datasets["test"]["labels"]))

# Convert the evaluation result into a pandas DataFrame for better readability
report_df = pd.DataFrame(evaluation_result).transpose()

# Save the report to a text file in a readable format
with open(REPORT_FILE, "w") as f:
    f.write("Evaluation Results:\n")
    report_df_str = report_df.to_string()
    f.write(report_df_str)


# Optionally, also print to the console
print("✅ Evaluation Results:")
print(report_df)
