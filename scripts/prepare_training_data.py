# scripts/prepare_training_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/reviews_analysis.csv"
TRAIN_OUT = "data/train_data.csv"
VAL_OUT = "data/val.csv"   # set for model evaluation

# Three-class mapping: NEGATIVE=0, POSITIVE=1, MIXED=2
LABEL_MAP = {
    "NEGATIVE": 0,
    "POSITIVE": 1,
    "MIXED": 2
}

def main():
    # Load & clean
    df = pd.read_csv(INPUT)
    df = df[["Text", "comprehend_sentiment"]].dropna()
    df["label"] = df["comprehend_sentiment"].map(LABEL_MAP).astype(int)

    # Before split — see raw distribution
    print("Class counts before split:")
    print(df["comprehend_sentiment"].value_counts())

    # Stratify so all classes keep their proportions
    train, val = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # Save
    train.to_csv(TRAIN_OUT, index=False)
    val.to_csv(VAL_OUT, index=False)

    # After split — confirm distribution
    print("\nClass counts after split — Train:")
    print(train["comprehend_sentiment"].value_counts())
    print("\nClass counts after split — Val:")
    print(val["comprehend_sentiment"].value_counts())

    print(f"\n✅ Wrote {TRAIN_OUT} and {VAL_OUT}")

if __name__ == "__main__":
    main()
