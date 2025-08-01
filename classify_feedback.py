
import boto3
import csv

# Initialize Comprehend client
comprehend = boto3.client('comprehend', region_name='us-east-1')

# Open and read the CSV file
with open('reviews.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        review = row.get('review_text') or row.get('Review Text')  # Handle different column headers
        if review and review.strip():
            response = comprehend.detect_sentiment(Text=review, LanguageCode='en')
            print(f"\nReview: {review}\nSentiment: {response['Sentiment']}\n")

with open('classified_reviews.csv', 'w', newline='', encoding='utf-8') as csvfile_out:
    fieldnames = ['Review', 'Sentiment']
    writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
    writer.writeheader()

    with open('reviews.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            review = row.get('review_text') or row.get('Review Text')
            if review and review.strip():
                response = comprehend.detect_sentiment(Text=review, LanguageCode='en')
                writer.writerow({'Review': review, 'Sentiment': response['Sentiment']})

