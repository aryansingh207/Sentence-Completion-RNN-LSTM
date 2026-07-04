import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("sentence_completion_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}

max_len = 749

def generate_text(seed_text, next_words):

    for _ in range(next_words):

        sequence = tokenizer.texts_to_sequences([seed_text])[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_len - 1,
            padding="pre"
        )

        predicted_id = np.argmax(
            model.predict(sequence, verbose=0),
            axis=-1
        )[0]

        output_word = index_to_word.get(
            predicted_id,
            ""
        )

        seed_text += " " + output_word

    return seed_text

st.title("Sentence Completion using LSTM")

user_input = st.text_input(
    "Enter starting text:"
)

num_words = st.slider(
    "Words to Generate",
    1,
    20,
    10
)

if st.button("Generate"):

    result = generate_text(
        user_input,
        num_words
    )

    st.success(result)