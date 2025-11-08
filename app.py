import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model=load_model('next_word_lstm.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")

#DropDown with a list of texts which gets put into input_text if slected otherwise we can also input in the text
predicted_words_list = [
    "",  # Empty default option
    "To be or not to be",
    "Before my God I might",
    "Peace break thee off",
    "Look where it comes",
    "If thou hast any sound",
    "Stay and speak stop",
    "We do it wrong being"
]

col1, col2 = st.columns(2)

with col1:
    selected_text = st.selectbox("Select a predefined text:", predicted_words_list)

with col2:
    custom_text = st.text_input("Or enter your own text:", "")

# Use either selected text or custom text
input_text = custom_text if custom_text else selected_text

# Only enable prediction if there is text input
button_disabled = not bool(input_text)
if st.button("Predict Next Word", disabled=button_disabled):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')