import json
import boto3
import pandas as pd
import streamlit as st
from io import StringIO
from datetime import datetime

# 🔧 Config – adjust once here and it's reflected everywhere
REGION = "us-east-1"
ENDPOINT = "sagemaker-scikit-learn-2025-08-15-13-46-04-474"   # must match your deployed SageMaker endpoint
BUCKET = "grey-customer-feedback-bucket"  # matches S3 bucket in your training/deployment
LOG_GROUP = "/ai-feedback-analyzer/predictions"
LOG_STREAM = "streamlit-app"

# 🏷️ Sentiment Mapping (for numeric predictions)
label_map = {
    0: "negative",
    1: "positive",
    2: "mixed"
}

# 🖼️ Page Setup
st.set_page_config(page_title="AI Feedback Analyzer", layout="wide")
st.title("🤖 AI Customer Feedback Analyzer")
st.markdown(
    "<h4 style='text-align: left;'>Powered by Amazon Comprehend, SageMaker, S3, CloudWatch, Streamlit UI</h4>",
    unsafe_allow_html=True
)

# 🔌 AWS Clients
rt = boto3.client("sagemaker-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
logs = boto3.client("logs", region_name=REGION)

# 📜 Ensure Log Group & Stream Exist
def ensure_log_resources():
    try:
        logs.create_log_group(logGroupName=LOG_GROUP)
    except logs.exceptions.ResourceAlreadyExistsException:
        pass
    try:
        logs.create_log_stream(logGroupName=LOG_GROUP, logStreamName=LOG_STREAM)
    except logs.exceptions.ResourceAlreadyExistsException:
        pass

ensure_log_resources()

# 📝 Log to CloudWatch
def log_to_cloudwatch(message):
    ts = int(datetime.now().timestamp() * 1000)
    try:
        logs.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=LOG_STREAM,
            logEvents=[{
                'timestamp': ts,
                'message': json.dumps(message)
            }]
        )
    except logs.exceptions.InvalidSequenceTokenException as e:
        token = e.response['expectedSequenceToken']
        logs.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=LOG_STREAM,
            sequenceToken=token,
            logEvents=[{
                'timestamp': ts,
                'message': json.dumps(message)
            }]
        )

# 🔍 Inference Function
def predict(text: str):
    payload = json.dumps({"text": text})
    resp = rt.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=payload
    )
    raw = json.loads(resp["Body"].read().decode("utf-8"))

    # ✅ Handle both numeric IDs and string labels from endpoint
    first_val = raw[0]
    if isinstance(first_val, str):
        try:
            # if it's a numeric string like "2"
            pred_class = int(first_val)
            sentiment = label_map.get(pred_class, "mixed")
        except ValueError:
            # it's already a label string like "positive"
            sentiment = first_val
    else:
        sentiment = label_map.get(int(first_val), "mixed")

    # 📡 Log prediction event
    log_to_cloudwatch({
        "event": "prediction",
        "text": text,
        "predicted_sentiment": sentiment,
        "timestamp": datetime.utcnow().isoformat()
    })

    return sentiment

# 🧭 Tabs
tab1, tab2 = st.tabs(["📤 Batch Upload", "🗣️ Single Feedback"])

# 📤 Tab 1: Batch Upload
with tab1:
    st.subheader("Upload CSV")
    up = st.file_uploader("CSV must contain a 'text' column", type=["csv"])
    if up:
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("Missing 'text' column.")
        else:
            st.write("📄 Preview", df.head())
            if st.button("🔍 Run Predictions"):
                with st.spinner("Analyzing..."):
                    df["sentiment"] = df["text"].fillna("").apply(predict)
                st.success("✅ Analysis Complete")
                st.dataframe(df[["text", "sentiment"]].head())

                # 📊 Label Distribution
                st.bar_chart(df["sentiment"].value_counts())

                # 💾 Save to S3
                csv_buf = StringIO()
                df.to_csv(csv_buf, index=False)
                if st.button("📦 Save to S3"):
                    key = "predictions/streamlit_predictions.csv"
                    s3.put_object(
                        Bucket=BUCKET,
                        Key=key,
                        Body=csv_buf.getvalue().encode("utf-8")
                    )
                    st.success(f"Saved to `s3://{BUCKET}/{key}`")

                    log_to_cloudwatch({
                        "event": "save_to_s3",
                        "bucket": BUCKET,
                        "key": key,
                        "rows_saved": len(df),
                        "timestamp": datetime.utcnow().isoformat()
                    })

# 🗣️ Tab 2: Single Input
with tab2:
    st.subheader("Analyze Single Feedback")
    txt = st.text_area("✍️ Enter feedback text", height=150)
    if st.button("🔍 Predict"):
        if not txt.strip():
            st.warning("Please enter some text.")
        else:
            sentiment = predict(txt)
            st.success(f"Sentiment: {sentiment}")
