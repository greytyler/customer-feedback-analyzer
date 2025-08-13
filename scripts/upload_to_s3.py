# scripts/upload_to_s3.py
import boto3, os
from botocore.exceptions import ClientError

REGION = "us-east-1"
BUCKET = "grey-customer-feedback-bucket"
LOCAL_PATH = "data/reviews.csv"
S3_KEY = "input/reviews.csv"

def main():
    s3 = boto3.client("s3", region_name=REGION)
    if not os.path.exists(LOCAL_PATH):
        raise FileNotFoundError(f"Missing file: {LOCAL_PATH}")
    try:
        s3.upload_file(LOCAL_PATH, BUCKET, S3_KEY)
        print(f"✅ Uploaded {LOCAL_PATH} → s3://{BUCKET}/{S3_KEY}")
    except ClientError as e:
        print("Upload failed:", e)

if __name__ == "__main__":
    main()

