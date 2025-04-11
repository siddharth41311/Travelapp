
import numpy as np
import pandas as pd
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
import string 
positive_reviews = pd.read_csv('hotel_positive_reviews.csv')
neutral_reviews = pd.read_csv('hotel_neutral_reviews.csv')
negative_reviews = pd.read_csv('hotel_negative_reviews.csv') 
positive_reviews.rename(columns={'positive reviews':'Reviews'},inplace=True)
neutral_reviews.rename(columns={'neutral reviews':'Reviews'},inplace=True)
negative_reviews.rename(columns={'negative reviews':'Reviews'},inplace=True) 
merged_df = pd.concat([positive_reviews,neutral_reviews,negative_reviews],ignore_index=True) 
print(merged_df)
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords') 
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words) 
merged_df['cleaned_review'] = merged_df['Reviews'].apply(preprocess_text) 
X_train, X_test, y_train, y_test = train_test_split(
    merged_df['cleaned_review'],  # Processed text data
    merged_df['sentiment'],    # Target labels
    test_size=0.2, random_state=42    # 80% Train, 20% Test
)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test) 
model = LogisticRegression()
model.fit(X_train_tfidf,y_train) 
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"TF-IDF + Logistic Regression Accuracy: {accuracy:.4f}")