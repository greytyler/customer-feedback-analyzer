This project is part of my hands-on journey to becoming a Cloud & AI Engineer. The AI Feedback Classifier uses AWS Comprehend and Python to analyze customer reviews and classify their sentiment as Positive, Negative, Neutral, or Mixed.

It’s a beginner-friendly, CLI-driven project built using only the terminal, AWS CLI, and Boto3 — no web apps or notebooks required.

---

What It Does

- Takes in a CSV file of customer feedback (e.g., product reviews)
- Uses Amazon Comprehend to detect the sentiment of each review
- Prints the sentiment result in the terminal
- Optionally writes results to a new CSV

---

Tech Stack

- Language: Python 3
- Cloud Service: AWS Comprehend (via Boto3)
- Tools: AWS CLI, Boto3, Pandas
- File Format: CSV

---

How to Run This Project

1. Clone the Repo

```bash
git clone https://github.com/YOUR-USERNAME/ai-feedback-classifier.git
cd ai-feedback-classifier

2. Set Up Your Environment
Install required packages:
pip install boto3 pandas

3. Configure AWS CLI
aws configure
Use IAM credentials for a user with this minimum access:
* AmazonComprehendFullAccess
* CloudWatchLogsFullAccess (optional)

4. Run the Script
Make sure your CSV is named reviews.csv or update the filename in classify_feedback.py.
Then run:
python3 classify_feedback.py
You'll see output like:
Review: I loved the fit and fabric!
Sentiment: POSITIVE

Dataset
The dataset used in this project was downloaded from Kaggle.com. I created a free Kaggle account and selected a customer reviews dataset that includes a column with customer-written feedback. You can use any dataset with a review_text or Review Text column.

Features
* 100% CLI and code-based workflow
* Beginner-friendly AWS Comprehend integration
* Modular: swap out datasets or extend functionality
* No AWS SDK wizardry — just clean, readable Python

Why I Built This
I'm learning AWS by building real-world projects — not just watching videos or reading docs. This is one of the first projects in my "Cloud Build Chronicles" series, where I document my growth, share code, and help demystify tech for other career changers.

Next Steps
* Connect to an S3 bucket for storing input/output files
* Add a simple Streamlit UI or API for live input
* Automate deployments with Terraform or CDK
* Include this project in a larger sentiment analytics dashboard

Connect with Me
Want to follow my cloud journey or see more beginner-friendly AWS projects?
YouTube: Des in the CloudPortfolio: desinthecloud.comInstagram + TikTok: @desinthecloud

Built with curiosity, courage, and a command line.
---