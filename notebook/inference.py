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

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
    return (model, vectorizer)

def input_fn(request_body, request_content_type):
    return request_body  # or parse JSON if needed

def predict_fn(input_data, model_bundle):
    model, vectorizer = model_bundle
    transformed = vectorizer.transform([input_data])
    prediction = model.predict(transformed)
    return prediction.tolist()
