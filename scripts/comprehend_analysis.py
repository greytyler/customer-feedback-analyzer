import boto3
import pandas as pd
import math

REGION = "us-east-1"
INPUT = "data/reviews.csv"
OUTPUT = "data/reviews_analysis.csv"
BUCKET = "grey-customer-feedback-bucket"
S3_KEY = "processed/reviews_analysis.csv"
LANG = "en"

def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def main():
    # Load input CSV
    df = pd.read_csv(INPUT)
    texts = df["Text"].fillna("").tolist()

    # Initialize Comprehend
    comp = boto3.client("comprehend", region_name=REGION)

    sentiments = []
    key_phrases_all = []

    for batch in chunks(texts, 25):
        # Sentiment
        s_resp = comp.batch_detect_sentiment(TextList=batch, LanguageCode=LANG)
        sentiments.extend([r.get("Sentiment") for r in s_resp["ResultList"]])

        # Key Phrases
        kp_resp = comp.batch_detect_key_phrases(TextList=batch, LanguageCode=LANG)
        for item in kp_resp["ResultList"]:
            phrases = [kp["Text"] for kp in item.get("KeyPhrases", [])]
            key_phrases_all.append(", ".join(sorted(set(phrases))))

    # Handle edge cases
    while len(sentiments) < len(df): sentiments.append("NEUTRAL")
    while len(key_phrases_all) < len(df): key_phrases_all.append("")

    # Add results to DataFrame
    df["comprehend_sentiment"] = sentiments
    df["comprehend_key_phrases"] = key_phrases_all

    # Save locally
    df.to_csv(OUTPUT, index=False)
    print(f"✅ Saved locally: {OUTPUT}")

    # Upload to S3
    s3 = boto3.client("s3")
    s3.upload_file(OUTPUT, BUCKET, S3_KEY)
    print(f"🚀 Uploaded to S3: s3://{BUCKET}/{S3_KEY}")

if __name__ == "__main__":
    main()
