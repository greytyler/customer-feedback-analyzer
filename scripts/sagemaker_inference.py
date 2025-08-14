# scripts/sagemaker_inference.py
import boto3, json, csv, os
import pandas as pd
import joblib

REGION = "us-east-1"
ENDPOINT = "feedback-analyzer-endpoint"
INPUT = "data/feedback_samples.csv"
OUTPUT = "data/model_predictions.csv"
BUCKET = "grey-customer-feedback-bucket"

USE_LOCAL_FALLBACK = True
UPLOAD_TO_S3 = True

def load_local_model():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

def predict_local(texts, model_bundle):
    model, vectorizer = model_bundle
    transformed = vectorizer.transform(texts)
    preds = model.predict(transformed)
    return [{"label": int(pred), "proba": None} for pred in preds]

def predict_remote(texts):
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
            result = json.loads(resp["Body"].read().decode("utf-8"))
            if "label" in result:
                results.append(result)
            else:
                results.append({"error": "Invalid response", "raw": result})
        except Exception as e:
            results.append({"error": str(e)})
    return results

def main():
    df = pd.read_csv(INPUT)
    texts = df["text"].fillna("").tolist()

    try:
        print("🔗 Sending to SageMaker endpoint...")
        preds = predict_remote(texts)
    except Exception as e:
        print(f"⚠️ Endpoint failed: {e}")
        if USE_LOCAL_FALLBACK:
            print("🔄 Falling back to local model...")
            model_bundle = load_local_model()
            preds = predict_local(texts, model_bundle)
        else:
            raise

    df["model_raw_result"] = preds
    df.to_csv(OUTPUT, index=False)
    print(f"✅ Saved predictions → {OUTPUT}")

    if UPLOAD_TO_S3:
        s3 = boto3.client("s3", region_name=REGION)
        s3.upload_file(OUTPUT, BUCKET, "predictions/model_predictions.csv")
        print(f"☁️ Uploaded to s3://{BUCKET}/predictions/model_predictions.csv")

if __name__ == "__main__":
    main()
