import argparse, os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train/train_data.csv")
    parser.add_argument("--val", type=str, default="/opt/ml/input/data/val/val.csv")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    return parser.parse_args()

def main():
    args = parse_args()
    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    X_train, y_train = train["Text"].astype(str), train["label"].astype(int)
    X_val, y_val = val["Text"].astype(str), val["label"].astype(int)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=20000)
    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)

    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, y_train)

    acc = clf.score(Xva, y_val)
    print({"validation_accuracy": float(acc)})

    joblib.dump(vectorizer, os.path.join(args.model_dir, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

if __name__ == "__main__":
    main()