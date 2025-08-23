### 🧠AI Customer feedback classifier

<img src="./screenshots/csv upload dark.png">

An end-to-end AI Customer feedback analyzer built using **Amazon Comprehend** for sentiment classification on a large diverse dataset, **SageMaker AI** for model training and deployment, **S3 storage**, **CloudWatch** for detailed AI prediction logs and **Streamlit App UI** using **Boto3** and custom **Python Code**. It analyzes customer reviews and classify their sentiment as Positive, Negative, Neutral, or Mixed.



---

##### What It Does

- Takes in a CSV file of customer feedback (e.g., product reviews)
- Uses Amazon Comprehend to detect the sentiment of each review
- Prints the sentiment result in the terminal
- Optionally writes results to a new CSV

---

##### Tech Stack

- Language: Python 3
- Cloud Service:Amazon SageMaker AI, AWS Comprehend (via Boto3), CloudWatch
- Tools: AWS CLI,VSCode terminal Boto3, Pandas, Matplotlib, Seaborn, Streamlit UI frontend
- File Format: CSV

---

How to Run This Project

1. Clone the Repo

```bash
git clone https://github.com/YOUR-USERNAME/customer-feedback-analyzer.git
cd customer-feedback-analyzer

2. Set Up Your Environment
Install required packages:
pip install

3. Configure AWS CLI or local development environment
aws configure
Use IAM credentials for a user with this minimum access.

4. Run the Script
Make sure your CSV is named reviews.csv or update the filename in comprehend_analysis.py.
Then run:
python3 comprehend_analysis.py
You'll see output like:
Review: I loved the fit and fabric!
Sentiment: POSITIVE

Dataset
The dataset used in this project was downloaded from Kaggle.com. I created a free Kaggle account and selected a customer reviews dataset that includes a column with customer-written feedback. You can use any dataset with a review_text or Review Text column.

Features
* 100% terminal, modular and code-based workflow
* Modular: swap out datasets or extend functionality
* Just clean, readable Python

Why I Built This
I'm learning more about AWS by building real-world projects — not just watching videos or reading docs. This is one of the first projects in my "Cloud Build Chronicles" series, where I document my growth, share code, and help demystify tech for other career changers.

Next Steps
* Automate deployments with Terraform or CDK
* Include this project in a larger sentiment analytics dashboard

Connect with Me
Want to follow my cloud journey or see more beginner-friendly AWS projects?
YouTube: Des in the CloudPortfolio: desinthecloud.comInstagram + TikTok: @desinthecloud

Built with curiosity, courage, and a command line.
---