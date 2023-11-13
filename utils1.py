#genreal system libriers
import random
import json
import pickle
import warnings
import traceback

#Pre_processing
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer

#Model
import sklearn
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold

# GUI
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk  # Import ttk for enhanced button styles

# from GUI import root

stemmer = LancasterStemmer()

global model_path , pickel_files_path,Dataset_path
model_path = './models/'
pickel_files_path = './pickle_files/'
Dataset_path = './Dataset/intents.json'


# Loading json dataset
def load_data(Dataset_path):
    with open(Dataset_path) as json_data: 
        return json.load(json_data)
    
    
def organize_Data(intents):
    
    words = []                  # store words that make tokinize for it 
    classes = []                # store tags 
    documents = []
    ignore_words = ['?']
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    # remove duplicates
    classes = sorted(list(set(classes)))

    print (len(documents), "documents")
    print (len(classes), "classes", classes)
    print (len(words), "unique stemmed words", words)
    
    return words,classes,documents


def generate_dataset(words,classes,documents):#(organize_Data(intents))
    # create our training data
    training = []
    output = []
    # create an empty array for our output
    output_empty = [0] * len(classes)

    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)

    # create train and test lists
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    return train_x,train_y


def train_and_save_model(pickel_files_path, words, classes, train_x, train_y):
    data = {'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}
    pickle.dump(words, open(str(pickel_files_path) + 'words.pkl', 'wb'))
    pickle.dump(classes, open(str(pickel_files_path) + 'classes.pkl', 'wb'))
    pickle.dump(data, open(str(pickel_files_path) + 'training_data.pkl', 'wb'))
    pickle.dump(train_x, open(str(pickel_files_path) + 'train_x.pkl', 'wb'))
    pickle.dump(train_y, open(str(pickel_files_path) + 'train_y.pkl', 'wb'))

    # Perform cross-validation
    num_folds = 2
    kf = KFold(n_splits=num_folds, shuffle=True)

    for train_index, test_index in kf.split(train_x):
        train_x_fold, test_x_fold = np.array(train_x)[train_index], np.array(train_x)[test_index]
        train_y_fold, test_y_fold = np.array(train_y)[train_index], np.array(train_y)[test_index]

        model = build_model(train_x, train_y)
        model.fit(train_x_fold, train_y_fold, epochs=200, batch_size=8, verbose=1)

    return model
                

def build_model(train_x, train_y):
    tf.keras.backend.clear_session()
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(len(train_x[0]),)))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(len(train_y[0]), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model




def classify_and_respond(input_text,model,words,classes,response_text,root,event=None):
    user_input = input_text.get("1.0", "end-1c")
    input_text.delete("1.0", tk.END)
    if user_input.lower() == 'quit':
        # root.title("Medical ChatBot")
        root.quit()
        root.quit()
        return 
    results = classify(user_input,model,words,classes)
    if results:
        intents = load_data(Dataset_path) 
        for intent in intents['intents']:
            if intent['tag'] == results[0][0]:
                response = np.random.choice(intent['responses'])
                update_chat("You: " + user_input, "ChatBot: " + response, "response",response_text)
                
    else:
        update_chat("You: " + user_input, "ChatBot: I'm sorry, but I don't understand. Can you please rephrase?", "error",response_text)


def classify(sentence,model,words,classes):
    # Generate probabilities from the model
    results = model.predict(np.array([bow(sentence, words)]))[0]
    # Filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > 0.25]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


def bow(sentence, words, show_details=False):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return np.array(bag)


def clean_up_sentence(sentence):
    # Tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # Stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def update_chat(user_message, bot_message, tag,response_text):
    response_text.insert(tk.END, user_message + "\n", tag)
    response_text.insert(tk.END, bot_message + "\n", tag)
    response_text.see(tk.END)

