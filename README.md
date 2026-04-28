# Customer-Sentiment-NLP
Classifies customer reviews as positive or negative using TF-IDF and Logistic Regression.

# Customer Sentiment Analysis Using NLP

This project uses supervised learning to classify customer reviews as positive or negative based on review text.

## Project Overview

The goal of this project is to predict customer satisfaction from text reviews. I used a Kaggle customer review dataset that includes review text and star ratings.

Ratings of 4 or 5 were labeled as positive, and ratings of 1 or 2 were labeled as negative. Neutral 3-star reviews were removed to make the classification task clearer.

## Methods Used

- Python
- Pandas
- TF-IDF Vectorization
- Logistic Regression
- Accuracy, Precision, Recall, and F1 Score
- Confusion Matrix

## Results

Accuracy: 0.924  
Precision: 0.934  
Recall: 0.908  
F1 Score: 0.957  

Overall, the model performed well, especially on positive reviews, but it struggled more with negative or mixed reviews.

## Limitations

The model can struggle with sarcasm, mixed sentiment, short reviews, and class imbalance.

## Dataset

Dataset source: https://www.kaggle.com/datasets/abdallahwagih/amazon-reviews
