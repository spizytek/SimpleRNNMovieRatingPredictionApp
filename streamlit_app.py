# Step 1: Import libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import streamlit as st


# load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


# load model
model = load_model('simplernn_imdb_model.h5')
# model.summary()


# Step 2: Helper functions
# for decoding reviews
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])


def preprocess_text(text):
    # Preprocess the review

    words = text.lower().split()
    max_words = 500
    # This is important: Keras reserves indices 0-3 for special tokens
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # Then pad the review to max length
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_words)
    return padded_review


# Step 3: Helper functions
# Create a function to make predictions
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    
    prediction = model.predict(processed_review)
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    return sentiment, prediction[0][0]


# Step 4: Streamlit App
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")    
user_input = st.text_area("Movie Review", "Type your review here...")   

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.write("Please enter a valid movie review.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: **{sentiment}** (Confidence: {confidence:.4f})")



