import os
from typing import Counter
from urllib import response
import nltk
import ssl
import streamlit as st
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

intents = [
    {
        "tag" : "pembuka",
        "patterns" : ["Hai", "Hallo", "Bagaimana kabarmu ?"],
        "responses" : ["Hai", "Hallo juga", "Alhamdulillah baik baik saja, terima kasih"]
    },

    {
        "tag" : "penutup",
        "patterns" : ["Sampai jumpa", "Dadah"],
        "responses" : ["Anytime"]
    },

    {
        "tag" : "terima kasih",
        "patterns" : ["Terima kasih", "Thanks"],
        "responses" : ["Masama", "No problem", "Ur welcome"]
    },

    {
        "tag" : "kuliner",
        "patterns" : ["Kuliner di salatiga apa saja ?", "Apa itu Sate Sapi Suruh ?", "Dimana Sate Sapi Suruh berada ?", "Soto Esto apa itu ?", "Soto Esto berada dimana ?"],
        "responses" : ["Kuliner di Salatiga ada Sate Sapi Suruh, Soto Esto, Gecok Kambing, Sate Blotongan, Warung Joglo Bu Rini, dll"]
    }
]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents :
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text) :
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents :
        if intent['tag'] == tag :
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main() :
    global counter
    st.title("Chatbot")
    st.write(
        "Welcome to Chatbot Kuliner Salatiga."
    )

    counter += 1
    user_input = st.text_input("You :", key = f"user_input_{counter}")

    if user_input :
        response = chatbot(user_input)
        st.text_area("Chatbot :", value = response, height = 100,
                     max_chars = None, key = f"chatbot_response_{counter}")
        print(counter)

        if response.lower() in ["Sampai jumpa", "Dadah"]:
            st.write("Terima kasih... Have a nice day :) ")
            st.stop()


if __name__ == '__main__':
    main()