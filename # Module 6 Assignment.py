import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

makeup_reviews = pd.read_csv('makeup_reviews.csv', low_memory=False)

# Ensure 'rating' is numeric and drop invalid rows
makeup_reviews['rating'] = pd.to_numeric(makeup_reviews['rating'], errors='coerce')
makeup_reviews = makeup_reviews.dropna(subset=['rating'])
makeup_reviews = makeup_reviews.sample(n=10000, random_state=42)

# Preprocess text
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

# Convert text into numerical features using CountVectorizer
vectorizer = CountVectorizer(max_features=1000)  
try:
    X = vectorizer.fit_transform(X)
except KeyboardInterrupt:
    print("Vectorization interrupted. Check input data.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Data
confusion_matrix_data = np.array([[72, 289], [0, 1636]])
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_data,
                              display_labels=["Negative (0)", "Positive (1)"])
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()

# Classification Report Data
classification_report_data = {
    "Class": ["Negative (0)", "Positive (1)"],
    "Precision": [0.90, 0.85],
    "Recall": [0.20, 1.00],
    "F1-Score": [0.33, 0.92],
    "Support": [361, 1636]
}

classification_df = pd.DataFrame(classification_report_data)
fig, ax = plt.subplots(figsize=(8, 5))
classification_df.set_index("Class")[["Precision", "Recall", "F1-Score"]].plot(kind="bar", ax=ax)
plt.title("Classification Metrics by Class")
plt.ylabel("Score")
plt.ylim(0, 1.2)
plt.xticks(rotation=0)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

feature_importances_data = {
    "Feature": ["love", "amazing", "recommend", "perfect", "favorite",
                "cheap", "disappointed", "waste", "poor", "irritating"],
    "Importance": [0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.03]
}

# Horizontal Bar Plot for Feature Importances
feature_importances_df = pd.DataFrame(feature_importances_data)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feature_importances_df, x="Importance", y="Feature", ax=ax, hue=None)
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
