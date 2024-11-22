import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data
product_info = pd.read_csv('path_to/product_info.csv')
review_files = ['reviews_0-250.csv', 'reviews_250-500.csv', 'reviews_500-750.csv', 'reviews_750-1250.csv', 'reviews_1250-end.csv']
reviews = pd.concat([pd.read_csv(f'path_to/{file}') for file in review_files])

# Merge data
data = pd.merge(reviews, product_info, on='product_id')

# Text preprocessing
def compute_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0

data['sentiment'] = data['review_text'].apply(compute_sentiment)

# Feature engineering
features = data.groupby('product_id').agg({
    'sentiment': 'mean',
    'helpful_votes': 'sum',
    'star_rating': 'mean',
    'price': 'mean'
}).reset_index()

# Create labels
features['popularity'] = (features['helpful_votes'] >= features['helpful_votes'].quantile(0.8)).astype(int)

# Split data
X = features[['sentiment', 'helpful_votes', 'star_rating', 'price']]
y = features['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
