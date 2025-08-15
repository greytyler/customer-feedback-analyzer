"""
inference.py

This script defines how SageMaker should:
- Load the trained model and vectorizer from the model.tar.gz file
- Handle incoming data from the endpoint
- Transform input and return predictions
It is used during endpoint deployment, not during training.
"""
import joblib
import os
import json

def model_fn(model_dir):
    print(f"[model_fn] Loading artifacts from: {model_dir}")

    model_path = os.path.join(model_dir, "model.joblib")
    vec_path = os.path.join(model_dir, "vectorizer.joblib")
    mapping_path = os.path.join(model_dir, "label_mapping.json")

    # Check for existence and log results
    for path in [model_path, vec_path]:
        if not os.path.exists(path):
            print(f"[model_fn][ERROR] Missing required file: {path}")
            raise FileNotFoundError(f"Required artifact not found: {path}")
        else:
            print(f"[model_fn] Found: {path}")

    model = joblib.load(model_path)
    print("[model_fn] Model loaded successfully")
    vectorizer = joblib.load(vec_path)
    print("[model_fn] Vectorizer loaded successfully")

    label_mapping = None
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            label_mapping = {int(k): v for k, v in json.load(f).items()}
        print("[model_fn] Label mapping loaded")
    else:
        print("[model_fn] No label mapping found — returning numeric predictions")

    return model, vectorizer, label_mapping

def input_fn(request_body, request_content_type):
    print(f"[input_fn] Received request with content_type: {request_content_type}")
    return request_body

def predict_fn(input_data, model_bundle):
    print(f"[predict_fn] Received input: {input_data}")
    model, vectorizer, label_mapping = model_bundle
    transformed = vectorizer.transform([input_data])
    prediction = model.predict(transformed)
    if label_mapping:
        result = [label_mapping[p] for p in prediction]
    else:
        result = prediction.tolist()
    print(f"[predict_fn] Returning prediction: {result}")
    return result
