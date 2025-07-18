import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib # Used for saving and loading the model

print("--- Training Script Started ---")

# 1. Load the Dataset
try:
    df = pd.read_csv('IMDB Dataset.csv')
    print("Dataset loaded.")
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found. Please place it in the same directory.")
    exit()

# 2. Prepare the Data
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
X = df['review']
y = df['sentiment']
print("Data prepared.")

# 3. Vectorize the Text
# We train the vectorizer on the ENTIRE dataset now since this is our final model
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)
print("Text vectorized.")

# 4. Train the Model
model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)
print("Model trained.")

# 5. Save the Vectorizer and the Model
# The vectorizer must be saved to process new text in the same way
joblib.dump(vectorizer, 'vectorizer.pkl')
# Save the trained model
joblib.dump(model, 'model.pkl')

print("\n--- Model and Vectorizer have been saved as 'model.pkl' and 'vectorizer.pkl' ---")

