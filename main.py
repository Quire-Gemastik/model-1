from typing import List
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import tensorflow as tf

keras = tf.keras
load_model = keras.models.load_model

model = load_model('model/data/modelFinal.h5', compile=False)
label_encoder = joblib.load('model/data/label_encoder.pkl')
tfidf_vectorizer = joblib.load('model/data/tfidf_vectorizer.pkl')
keywords = joblib.load('model/data/keywords.pkl')
feature_vec = joblib.load('model/data/feature_vec.pkl')
labels = joblib.load('model/data/labels.pkl')

app = FastAPI()

class UserInput(BaseModel):
    text: str

def modelFunction(user_input, tfidf_vectorizer):
    user_tfidf = tfidf_vectorizer.transform([user_input])

    predicted_probabilities = model.predict(user_tfidf.toarray())
    predicted_label = np.argmax(predicted_probabilities, axis=1)[0]
    predicted_label_string = label_encoder.inverse_transform([predicted_label])[0]

    input_keywords = set(word.lower() for word in user_input.split() if word.lower() in keywords.get(predicted_label_string, []))

    user_similarity = cosine_similarity(user_tfidf, feature_vec)[0]

    keyword_indices = [tfidf_vectorizer.vocabulary_.get(word) for word in input_keywords if word in tfidf_vectorizer.vocabulary_]

    if keyword_indices:
        keyword_mask = feature_vec[:, keyword_indices].sum(axis=1).A1 > 0  # Sum along columns and check if > 0
        user_similarity[keyword_mask] *= 1.5

    index_similar_texts = user_similarity.argsort()[::-1]

    # Select unique labels (only one label that is the same)
    unique_labels = set()
    recommended_labels = []
    similarities = []

    for idx in index_similar_texts:
        if len(recommended_labels) >= 5:  # Limit to only 5 recommendations
            break

        label = labels[idx]
        label_string = label_encoder.inverse_transform([label])[0]  # Convert back to label string
        if label_string not in unique_labels:
            unique_labels.add(label_string)
            recommended_labels.append(label_string)
            similarities.append(user_similarity[idx])

    return predicted_label, predicted_label_string, recommended_labels[:5], similarities[:5]

@app.post("/predict")
def predict(input: UserInput):
    user_input = input.text
    predicted_label, predicted_label_string, recommended_labels, similarities = modelFunction(user_input, tfidf_vectorizer)

    if not predicted_label_string:
        raise HTTPException(status_code=400, detail="Prediction error")

    return {
        "predicted_label": predicted_label_string,
        "recommended_labels": recommended_labels,
        "similarities": similarities
    }

@app.get("/")
def read_root():
    return {"Hello": "World"}