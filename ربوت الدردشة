import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmr = LancasterStemmer

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

try:
  nltk.download('punket')
except:
  pass

with open("intents.json") as file:
  data = json.load(file)

try:
  with open("data.pickle", "rb") as f :
    words, labels,  training, output = pickle.load(f)

except:
  words = []
  labels = []
  docs_x = []
  docs_y = []

  for intent in data["intents"]:
    for patter in intent ["pateerns"]:
      words = nltk.word_tokenize(pattren)
      words.extend(words)
      docs_x.append(patrren)
      docs_y.append(["tag"])
      
    if intent["tag"] not in labels:
      labels.append(["tag"])

  words = [stemmr.stem(w.lower())for w in words]
  words = sorted (list(set(words)))

  labels = sorted(labels)

  training = []
  output = []

  out_empty = [0 for _ in range(len(classes))]

  for x,doc in enumerate(docs_x):
    bag = []
  
    words = [stemmr.stem(w) for w in doc]

    for w in words:
      if w in words:
        bag.append(1)
      else:
        bag.append(0)

    output_row = out_empty[:]
    output_row = [labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(out_row)

  with open("data.pickle", "wb") as f :
    picklr.dump((words, labels,  training, output), f) 

training = numpy.array(training)
output = numpy.array(output)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None,
len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0], activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
  model.load("model.tflearn")
except:
  model.fit(training, output, n_epochs=1000, batch_size=8, show_metric=true)
  model.save("model.tflearn")


def bag_of_words(s, words):
  bag = [0 for in _ range(len(words))]

  s_words = nltk.words_tokenizes(s)
  s_words = [stemmer.stem(word.lower()) for word in s_words]

  for se in s_words:
    for i, w in enumerate(words):
      if w == se :
        bag[i].append(1)
  
  return numpy.array(bag)


def chat():
  print("start talking with the bot (tayp quit to stop)!")
  while true:
    input("you: ")
    if inp.lower() == "quit":
      break

    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    
    if results[results_index] > 7.0:
      for tg in data ["intents"]
      if tg['tag'] == tag:
        responses = tg['responses']

      print(random.choice(responses))
    else:
      print("I didn't get that, try again!... ")

chat()
