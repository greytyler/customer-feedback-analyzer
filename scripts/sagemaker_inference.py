# Load Input from S3 → Run SageMaker Predictions → Save/Upload Results

import boto3, json, os
import pandas as pd
import joblib

# --- Config ---
REGION = "us-east-1"
ENDPOINT = "sagemaker-scikit-learn-2025-08-15-13-46-04-474"
INPUT_KEY = "input/feedback_samples.csv"   # S3 key for input file
OUTPUT = "data/model_predictions.csv"
BUCKET = "grey-customer-feedback-bucket"

USE_LOCAL_FALLBACK = True
UPLOAD_TO_S3 = True

def load_local_model():
    """Load local model + vectorizer for offline predictions."""
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

def predict_local(texts, model_bundle):
    """Run predictions with the local model."""
    model, vectorizer = model_bundle
    transformed = vectorizer.transform(texts)
    preds = model.predict(transformed)
    return [{"label": str(pred), "proba": None} for pred in preds]

def parse_endpoint_result(result):
    """
    Normalizes the model output into a standard dict:
    - If endpoint returns list: take first item
    - If dict with 'label': use that
    - Else: return raw
    """
    if isinstance(result, dict):
        if "label" in result:
            return {"label": result["label"], "proba": result.get("proba")}
        return {"raw": result}
    elif isinstance(result, list):
        return {"label": result[0]} if result else {"error": "Empty list"}
    elif isinstance(result, str):
        return {"label": result}
    return {"error": "Unknown format", "raw": result}

def predict_remote(texts):
    """Send texts to SageMaker endpoint and parse results."""
    rt = boto3.client("sagemaker-runtime", region_name=REGION)
    results = []
    for txt in texts:
        payload = json.dumps({"text": txt})
        try:
            resp = rt.invoke_endpoint(
                EndpointName=ENDPOINT,
                ContentType="application/json",
                Body=payload
            )
            raw_result = json.loads(resp["Body"].read().decode("utf-8"))
            results.append(parse_endpoint_result(raw_result))
        except Exception as e:
            results.append({"error": str(e)})
    return results

def main():
    # --- Load test data from S3 ---
    print(f"☁️ Downloading s3://{BUCKET}/{INPUT_KEY} ...")
    s3 = boto3.client("s3", region_name=REGION)
    obj = s3.get_object(Bucket=BUCKET, Key=INPUT_KEY)
    df = pd.read_csv(obj["Body"])

    if "text" not in df.columns:
        raise ValueError(f"❌ '{INPUT_KEY}' in s3://{BUCKET} must have a 'text' column.")

    texts = df["text"].fillna("").tolist()

    # --- Try SageMaker endpoint, else local fallback ---
    try:
        print(f"🔗 Sending {len(texts)} records to SageMaker endpoint '{ENDPOINT}'...")
        preds = predict_remote(texts)
    except Exception as e:
        print(f"⚠️ Endpoint failed: {e}")
        if USE_LOCAL_FALLBACK:
            print("🔄 Falling back to local model...")
            model_bundle = load_local_model()
            preds = predict_local(texts, model_bundle)
        else:
            raise

    # --- Save results locally ---
    df["model_raw_result"] = preds
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"✅ Saved predictions → {OUTPUT}")

    # --- Optionally push to S3 ---
    if UPLOAD_TO_S3:
        s3.upload_file(OUTPUT, BUCKET, "predictions/model_predictions.csv")
        print(f"☁️ Uploaded to s3://{BUCKET}/predictions/model_predictions.csv")

if __name__ == "__main__":
    main()
