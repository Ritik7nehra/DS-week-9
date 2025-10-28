# train_sentiment.py
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Minimal training data (you can expand or replace with small dataset provided)
X_train = ["I love this", "This is great", "I hate this", "Terrible experience"]
y_train = [1, 1, 0, 0]

pipeline = make_pipeline(
    CountVectorizer(),
    LogisticRegression(solver="liblinear")
)
pipeline.fit(X_train, y_train)

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("Saved sentiment_model.pkl")
