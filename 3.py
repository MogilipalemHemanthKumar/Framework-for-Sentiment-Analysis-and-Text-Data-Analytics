import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
model = keras.models.load_model('model.h5')  # Load Keras HDF5 format
df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
data = df.head(1000)
sentences = data['Review Text'].astype(str).str.lower()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1
print(total_words)
tokenized_sentences = tokenizer.texts_to_sequences(data['Review Text'].astype(str))
input_sequences = []
for line in tokenized_sentences:
    for i in range(1, len(line)):
        n_gram_sequence = line[:i+1]
        input_sequences.append(n_gram_sequence)

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
def complete_this_paragraph(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text
x=st.button("Suggest")
st.write(complete_this_paragraph(x, 4))

