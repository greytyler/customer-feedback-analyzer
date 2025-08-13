# scripts/prepare_training_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/reviews_analysis.csv"
TRAIN_OUT = "data/train_data.csv"
VAL_OUT = "data/val.csv"   # set for model evaluation`

LABEL_MAP = {"POSITIVE":1, "NEGATIVE":0, "NEUTRAL":2, "MIXED":2}  # simple scheme

def main():
    df = pd.read_csv(INPUT)
    df = df[["Text", "comprehend_sentiment"]].dropna()
    df["label"] = df["comprehend_sentiment"].map(LABEL_MAP).fillna(2).astype(int)

    # keep 3 classes but you can also map to binary if preferred
    train, val = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train.to_csv(TRAIN_OUT, index=False)
    val.to_csv(VAL_OUT, index=False)
    print(f"✅ Wrote {TRAIN_OUT}, {VAL_OUT}")

if __name__ == "__main__":
    main()
