import os
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout # type: ignore
from keras.optimizers import SGD # type: ignore
import random

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Check if data.json exists
if not os.path.exists('data.json'):
    raise FileNotFoundError("The file 'data.json' is missing.")

# Load intents file
data_file = open('data.json').read()
intents = json.loads(data_file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        if pattern.strip() == "":
            continue
        
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))

        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word, remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Debugging logs
print(f"Words: {words}")
print(f"Classes: {classes}")
print(f"Documents: {documents[:5]}")  # Print the first 5 documents

# Save words and classes
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

# Create our training data
for doc in documents:
    # Initialize our bag of words
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle features and turn into Numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists. X - patterns, Y - intents
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# Debugging logs
print(f"Training data (X): {train_x[:5]}")
print(f"Training data (Y): {train_y[:5]}")

# Create the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model and save it
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('model.keras')

# Save the lemmatizer
pickle.dump(lemmatizer, open('lemmatizer.pkl', 'wb'))

print("Model created and saved successfully.")