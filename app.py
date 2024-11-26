import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st



# Encoder and decoder dictionaries
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key ,value in word_index.items()}


# Load the model
model = load_model("RNN_IMDB.keras")


def preprocess_text(text:str):
    words = text.split()

    # Note if word not found the word_index.get() 2 
    encoded_reviews = [word_index.get(word , 2) + 3 for word in words]

    # Pad the sequence 
    padded_review = pad_sequences([encoded_reviews] , maxlen=500 )

    return padded_review


def predict_sentiment(review):

    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)

    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return sentiment , prediction[0][0]



## Streamlit app

st.title("Sentiment Analysis")
st.write("Enter a review for sentiment Analysis")


# User input
user_input = st.text_input(label="Enter Review")

    
if st.button("Get Review") : 
    if user_input:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)

        sentiment_Score = prediction[0][0]
        sentiment = "Positive" if sentiment_Score > 0.5 else "Negative"


        # Diplay the result

        st.write(f"Sentiment :- {sentiment}")
        st.write(f"Sentiment Score :- {sentiment_Score}")

    else:
        st.write("Review cannot be blank")

