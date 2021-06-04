# -*- coding: utf-8 -*-
"""
Created on Sun Jul 04 14:52:43 2020

@author: Siddharth.S
"""


import numpy as np
import pickle
import pandas as pd
import tensorflow as tf

from PIL import Image
import json
import nltk
# Natural Language Toolkit
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# This stemmer object finds the root word of a word
# Eg
# program, programmer, programs, programming will all result in the same word program which is
# is the root of all the above words.
import tflearn
import random
import gradio as gr

with open("intents_med.json") as f:
  data = json.load(f)

with open("./modelDataMed/data_med.pickle", "rb") as f:
  words, labels, training, output = pickle.load(f)

tf.compat.v1.reset_default_graph()
# Removing the above causes an unexpected error, so just let it be 

net = tflearn.input_data(shape=[None, len(training[0])])    # The input layer of the DNN
net = tflearn.fully_connected(net, 8)                       # First hidden layer
net = tflearn.fully_connected(net, 8)                       # Second hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")   # Output layer
net = tflearn.regression(net)
# The above is explaining the system how to optimize the output of the DNN. The default
# is the adam optimizer.

model = tflearn.DNN(net)
model.load("./modelDataMed/model_med.tflearn", weights_only=True)

def createInput(userString):
    global words
    
    phrase = nltk.word_tokenize(userString)
    phrase = [stemmer.stem(word.lower()) for word in phrase]

    newEncoding = []
    for single_word in words:
        if single_word in phrase:
            newEncoding.append(1)
        else:
            newEncoding.append(0)
    return np.array(newEncoding)

def replyUser(query):
    result = model.predict([createInput(query)])
    result_index = np.argmax(result)
    # It returns an array of results. But, since in our case we only have one result, we can use 0 indexing to 
    # fetch the first result.

    FinalTag = labels[result_index]
    # FinalResponse = random.choice(data["intents"][tag]["patterns"]) is not possible since data["intents"]
    # return a list and not a dictionary
    if result[0][result_index]>0.7:
      for elements in data["intents"]:
        if FinalTag == elements["tag"]:
            FinalResponse = random.choice(elements["responses"])
            break
      return FinalResponse
    else:
        return "I didn't understand. Would you care to repeat?"


def main(query):
      
    Response=""
    Response=replyUser(query)
    return Response


gr.Interface(fn=main, inputs=gr.inputs.Textbox(default="What to do if someone has a cut?"), outputs=["textbox"]).launch()    