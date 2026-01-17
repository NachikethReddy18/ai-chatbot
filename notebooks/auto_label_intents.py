# -------------------------------
# Auto-label Twitter Support Data
# Zero-shot Intent Classification
# -------------------------------

import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from src.utils.preprocessing import clean_text

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/labeled_twitter_support.csv"
SAMPLE_SIZE = 50000
CONFIDENCE_THRESHOLD = 0.6

INTENTS = [
    "order_status",
    "refund_request",
    "shipping_issue",
    "payment_issue",
    "account_issue",
    "product_info",
    "complaint",
    "technical_issue",
    "contact_support"
]

# -------------------------------
# LOAD & PREPROCESS DATA
# -------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Keep only customer messages
df = df[df["inbound"] == True]
df = df[["text"]].dropna()

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

# Remove empty or very short texts
df = df[df["clean_text"].str.len() > 5]

# HARD LIMIT TO 50K ROWS
df = df.sample(n=SAMPLE_SIZE, random_state=42)

print(f"Running on {len(df)} rows")

# Safety check
assert len(df) <= SAMPLE_SIZE, "Dataset size exceeds limit!"

# -------------------------------
# LOAD ZERO-SHOT CLASSIFIER
# -------------------------------
print("Loading zero-shot model (this may take time)...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1  # CPU
)

# -------------------------------
# PREDICTION FUNCTION (SAFE)
# -------------------------------
def predict_intent(text):
    if not isinstance(text, str) or text.strip() == "":
        return "unknown", 0.0

    result = classifier(text, INTENTS)
    return result["labels"][0], result["scores"][0]

# -------------------------------
# AUTO-LABEL INTENTS
# -------------------------------
print("Auto-labeling intents...")
tqdm.pandas()

df[["intent", "confidence"]] = df["clean_text"].progress_apply(
    lambda x: pd.Series(predict_intent(x))
)

# Filter low-confidence predictions
df = df[df["confidence"] >= CONFIDENCE_THRESHOLD]

# -------------------------------
# SAVE OUTPUT
# -------------------------------
os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Step B completed successfully!")
print(f"Saved labeled dataset to: {OUTPUT_PATH}")
