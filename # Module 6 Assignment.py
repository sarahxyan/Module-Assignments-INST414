import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Load and preprocess data
makeup_reviews = pd.read_csv('makeup_reviews.csv', low_memory=False)

# Ensure 'rating' is numeric and drop invalid rows
makeup_reviews['rating'] = pd.to_numeric(makeup_reviews['rating'], errors='coerce')
makeup_reviews = makeup_reviews.dropna(subset=['rating'])
makeup_reviews = makeup_reviews.sample(n=10000, random_state=42)

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  
    return text

makeup_reviews['cleaned_text'] = makeup_reviews['review_text'].apply(preprocess_text)
makeup_reviews = makeup_reviews[makeup_reviews['cleaned_text'].str.strip() != '']

# Create binary sentiment labels
makeup_reviews['label'] = makeup_reviews['rating'].apply(lambda x: 1 if x >= 4 else 0)

X = makeup_reviews['cleaned_text']
y = makeup_reviews['label']


vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=["Negative (0)", "Positive (1)"])
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()


y_test = y_test.reset_index(drop=True)
test_indices = y_test.index

# Identify misclassified samples
misclassified_indices = [i for i in range(len(y_pred)) if y_pred[i] != y_test[i]]
misclassified_samples = makeup_reviews.iloc[misclassified_indices]
print("Misclassified Samples:\n", misclassified_samples)

# Visualize Feature Importances
feature_importances = model.feature_importances_
top_features = pd.DataFrame({
    'Feature': vectorizer.get_feature_names_out(),
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top_features, x='Importance', y='Feature', ax=ax)
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()
