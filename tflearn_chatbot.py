import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn as tfl
import tensorflow as tf
import json
stemmer = LancasterStemmer()

with open("intents.json") as file:
    intents = json.load(file)

stems = []
classes = []
tagged_words = []
ignore_words = ['?', '-', '.']

# Filtering through json data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        stems.extend(w)
        tagged_words.append((w, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent["tag"])

stems = [stemmer.stem(w.lower()) for w in stems if w not in ignore_words]
stems = sorted(list(set(stems)))
classes = sorted(list(set(classes)))

# print("tagged_words:", tagged_words)
# print("len:", len(stems), "stems:", stems)
# print("classes: ", classes)

training = []
output = []
output_empty = [0] * len(classes)

for match in tagged_words:
    bag = []
    pattern_word_stems = [stemmer.stem(word.lower()) for word in match[0]]
    for stem in stems:
        if stem in pattern_word_stems:
            bag.append(1)
        else:
            bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(match[1])] = 1
    training.append([bag, output_row])

# print("training:", training)
for x in range(len(tagged_words)):
    print(tagged_words[x][0], ":", training[x][0])
    print(tagged_words[x][1], ":", training[x][1])
    print('\n')
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# tf.compat.v1.reset_default_graph()
# net = tfl.input_data(shape=[None, len(train_x[0])])
# net = tfl.fully_connected(net, 8)
# net = tfl.fully_connected(net, 8)
# net = tfl.fully_connected(net, len(train_y[0]), activation='softmax')
# net = tfl.regression(net)

# model = tfl.DNN(net, tensorboard_dir='tflearn_logs')
# model.fit(train_x, train_y, n_epoch=567, batch_size=8, show_metric=True)
# model.save('chatbot.tflearn')
#
# with open('training_data', 'wb') as p:
#     pickle.dump({'stems': stems, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, p)