# 🏥 Health Care Chatbot

A simple AI-powered chatbot built using **Flask**, **NLTK**, and **Keras**, designed to assist users with basic health-related queries. This project demonstrates natural language processing, machine learning, and web development.

---

##  Project Structure

```
HealthCareChatbot/
│
├── app.py                 # Flask app to serve the chatbot
├── training.py            # Script to preprocess and train the model
├── data.json              # Intents: patterns and responses
├── model.keras            # Trained Keras model
├── texts.pkl              # Tokenized words (pickle)
├── labels.pkl             # Encoded labels (pickle)
│
├── templates/
│   └── index.html         # Chatbot frontend
│
├── static/
│   ├── styles/
│   │   └── style.css      # Custom CSS
│   └── images/
│       ├── bot.png        # Chatbot avatar
│       └── person.png     # User avatar
│
└── requirements.txt       # Python dependencies
```

---

##  Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/nandana134/health-care-chatbot.git
cd healthcare-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Resources

```python
import nltk
nltk.download('punkt')
```

### 4. Train the Chatbot

```bash
python training.py
```

### 5. Run the Flask App

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5001
```

---

##  Usage

* Ask health-related questions (e.g., *"I have a headache"*)
* The chatbot will respond with relevant suggestions or information
* If the input is unrecognized, it will return a fallback response

---


##  Example: `data.json`

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello! How can I help you today?"]
    },
    {
      "tag": "headache",
      "patterns": ["I have a headache", "My head hurts"],
      "responses": ["Try resting and staying hydrated. If it persists, consult a doctor."]
    }
  ]
}
```

---

## 🧠 Dependencies

* `Flask` - Web framework
* `nltk` - Natural Language Toolkit
* `keras` - Deep learning model
* `tensorflow` - Backend for Keras

---

## 🛠 Troubleshooting

* **Missing `data.json`**: Make sure it’s in the root directory.
* **Model not found**: Run `training.py` first to generate `model.keras`.
* **Unrecognized inputs**: Add more patterns under the `noanswer` intent in `data.json`.

---

## 🚀 Future Improvements

* Add more health-related intents
* Connect to a real-time medical database or API
* Deploy on AWS, Heroku, or Render
* Add voice recognition and text-to-speech

---

##  License

MIT License. Feel free to use, modify, and share this project.

---

##  Acknowledgments

* [NLTK](https://www.nltk.org/)
* [Keras](https://keras.io/)
* [TensorFlow](https://www.tensorflow.org/)
* [Flask](https://flask.palletsprojects.com/)

---


