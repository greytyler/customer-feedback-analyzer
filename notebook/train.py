import argparse
import os
import sys
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train/train_data.csv")
    parser.add_argument("--val", type=str, default="/opt/ml/input/data/val/val.csv")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return parser.parse_args()

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)  # BOM
        .str.strip()
        .str.lower()
    )
    return df

def main():
    args = parse_args()

    # Load
    train_df = pd.read_csv(args.train)
    val_df   = pd.read_csv(args.val)

    # Normalize headers and show them
    train_df = normalize_headers(train_df)
    val_df   = normalize_headers(val_df)
    print("🗂 Train columns:", train_df.columns.tolist())
    print("🗂 Val columns:", val_df.columns.tolist())

    # Columns (after normalization)
    text_col  = "text"
    label_col = "label"  # change to 'sentiment' etc. if needed

    # Validate presence in BOTH splits
    missing = []
    for name in (text_col, label_col):
        if name not in train_df.columns:
            missing.append(f"train missing '{name}'")
        if name not in val_df.columns:
            missing.append(f"val missing '{name}'")
    if missing:
        sys.exit("❌ Column mismatch: " + " | ".join(missing))

    # Clean rows
    train_df = train_df.dropna(subset=[text_col, label_col])
    val_df   = val_df.dropna(subset=[text_col, label_col])

    # Handle string labels by mapping to ints
    if not pd.api.types.is_integer_dtype(train_df[label_col]):
        uniq = sorted(train_df[label_col].astype(str).unique().tolist())
        label2id = {k: i for i, k in enumerate(uniq)}
        print("🔢 Label mapping:", label2id)
        train_df[label_col] = train_df[label_col].astype(str).map(label2id)
        # Ensure val uses same mapping
        if not pd.api.types.is_integer_dtype(val_df[label_col]):
            if set(val_df[label_col].astype(str)) - set(label2id.keys()):
                sys.exit("❌ val.csv contains unseen label values not present in train.csv")
            val_df[label_col] = val_df[label_col].astype(str).map(label2id)
    else:
        label2id = None

    X_train = train_df[text_col].astype(str)
    y_train = train_df[label_col].astype(int)
    X_val   = val_df[text_col].astype(str)
    y_val   = val_df[label_col].astype(int)

    # Basic sanity checks
    if len(set(y_train)) < 2:
        sys.exit("❌ Training labels contain fewer than 2 classes.")
    if len(set(y_val)) < 1:
        sys.exit("❌ Validation labels are empty.")

    print("📊 Train size:", len(X_train), "Val size:", len(X_val))
    print("📈 Train class counts:", train_df[label_col].value_counts().to_dict())
    print("📉 Val class counts:", val_df[label_col].value_counts().to_dict())

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)

    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train_tfidf, y_train)

    acc = clf.score(X_val_tfidf, y_val)
    print({"validation_accuracy": float(acc)})

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(args.model_dir, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
    if label2id is not None:
        with open(os.path.join(args.model_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(label2id, f)
    print("✅ Saved model artifacts to:", args.model_dir)

if __name__ == "__main__":
    main()
