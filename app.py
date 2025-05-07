import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np
import random
import json
import logging

from keras.models import load_model  # type: ignore
from flask import Flask, render_template, request

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load trained data and model
model = load_model('model.keras')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'

# Tokenize and lemmatize input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence into bag of words vector
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

# Predict intent
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
         return [{"intent": "noanswer", "probability": "1.0"}]
        

    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Generate bot response
def getResponse(ints, intents_json):
    if len(ints) == 0:
        return "Sorry, I couldn't understand that. Could you rephrase?"

    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Oops! Something went wrong."

# Chatbot response
def chatbot_response(msg):
    ints = predict_class(msg, model)
    logging.info(f"Predicted intents: {ints}")
    res = getResponse(ints, intents)
    return res

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    logging.info(f"User input: {userText}")
    response = chatbot_response(userText)
    logging.info(f"Bot response: {response}")
    return response

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5001)