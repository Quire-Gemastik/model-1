import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Dropout
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from collections import defaultdict
from nltk.corpus import stopwords
from collections import defaultdict
# from tensorflow.distribute import DistributedDatasetSpec
# from tensorflow.python.distribute import DistributedDatasetSpec

keras = tf.keras
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout


df = pd.read_csv("model/src/dataset.csv")

# Download stopwords and punkt if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Case Folding, Punctuation Removal, Stop Word Removal, Text Normalization/Noise Removal, and Stemming
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    # Case Folding
    text = text.lower()

    # Punctuation Removal
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    words = word_tokenize(text)

    # Stop Word Removal and Stemming
    words = [ps.stem(word) for word in words if word not in stop_words]

    return ' '.join(words)

# Apply preprocessing to each text in the dataframe
df['preprocessed_text'] = df['Cerita'].apply(preprocess_text)

# Collect unique words based on label
keywords = defaultdict(list)

for label, text in zip(df['Pekerjaan'], df['preprocessed_text']):
    words = set(text.split())
    keywords[label].extend(words)

for label in keywords:
    keywords[label] = [word for word in keywords[label] if word not in stop_words]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
feature_vec = vectorizer.fit_transform(df['preprocessed_text'].values)

# Label Encoding
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['Pekerjaan'].values)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(feature_vec,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=42)

joblib.dump(label_encoder, 'model/label_encoder.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
joblib.dump(feature_vec, 'model/feature_vec.pkl')
joblib.dump(labels, 'model/labels.pkl')
joblib.dump(keywords, 'model/keywords.pkl')

# Multi Layer Perceptron
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(set(labels)), activation='softmax')  # Number of output neurons = number of classes (majors)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 / 10**(epoch / 100))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(X_train.toarray(),
          y_train,
          epochs=20,
          batch_size=64,
          validation_data=(X_test.toarray(), y_test),
          callbacks=[lr_schedule])

# model.save('modelFinal.h5', custom_objects={'DistributedDatasetSpec': DistributedDatasetSpec})
model.save('model/modelFinal.h5')
model.save('model/modelFinal.keras')
joblib.dump(model, 'model/modelFinal.pkl')