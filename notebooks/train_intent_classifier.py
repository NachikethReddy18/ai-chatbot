# ----------------------------------
# Train DistilBERT Intent Classifier
# ----------------------------------

import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

import evaluate

# Ensure project root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "data/processed/labeled_twitter_support.csv"
MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = "models/intent_distilbert"

# -------------------------------
# LOAD DATA
# -------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Keep only required columns
df = df[["clean_text", "intent"]].dropna()

# -------------------------------
# INITIAL LABEL ENCODING
# -------------------------------
le = LabelEncoder()
df["label"] = le.fit_transform(df["intent"])

# -------------------------------
# REMOVE RARE CLASSES (< 2 samples)
# -------------------------------
label_counts = df["label"].value_counts()
valid_labels = label_counts[label_counts >= 2].index
df = df[df["label"].isin(valid_labels)]

# -------------------------------
# RE-ENCODE LABELS (IMPORTANT)
# -------------------------------
le = LabelEncoder()
df["label"] = le.fit_transform(df["intent"])

num_labels = len(le.classes_)
print(f"Number of intent classes after filtering: {num_labels}")

# -------------------------------
# SAVE LABEL MAP (FOR INFERENCE)
# -------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

label_map = {i: label for i, label in enumerate(le.classes_)}
pd.Series(label_map).to_json(
    os.path.join(OUT_DIR, "label_map.json")
)

# -------------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df["label"]
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

# -------------------------------
# TOKENIZATION
# -------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["clean_text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)

columns = ["input_ids", "attention_mask", "label"]
train_ds.set_format(type="torch", columns=columns)
val_ds.set_format(type="torch", columns=columns)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

# -------------------------------
# METRICS
# -------------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# -------------------------------
# TRAINING CONFIG
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    report_to="none"
)


# -------------------------------
# TRAINER
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -------------------------------
# TRAIN
# -------------------------------
print("Starting training...")
trainer.train()

# -------------------------------
# SAVE MODEL
# -------------------------------
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("✅ Step C completed successfully!")
print(f"Model saved to: {OUT_DIR}")
