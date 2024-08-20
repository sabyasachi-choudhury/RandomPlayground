import json
import random
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras import models, layers
from nltk import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# taking intents
with open("intents.json") as file:
    data = json.load(file)

patterns = []
ind_tags = []
tags = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        ind_tags.append(intent["tag"])
    if intent["tag"] not in tags:
        tags.append(intent["tag"])
num_classes = len(tags)

# initialize tokenizer
tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
# print(word_index)

# word tokenize
for i in range(len(patterns)):
    patterns[i] = word_tokenize(patterns[i])

# print(len(ind_tags), ind_tags)
# print(len(patterns), patterns)
for i in range(len(ind_tags)):
    out = [0] * num_classes
    out[tags.index(ind_tags[i])] = 1
    ind_tags[i] = out
# print(len(ind_tags), ind_tags)


# skip gram
def gram(seq, l=3):
    ret_seq = []
    for word in seq:
        if word in '!@#$%^&*()1234567890=_+[]{}|;"\',./?><':
            seq.remove(word)
    if len(seq) > l:
        for x in range(len(seq) - l + 1):
            ret_seq.append(seq[x:x+l])
        return ret_seq
    else:
        return [seq]


grams = []
gram_tags = []
for i in range(len(patterns)):
    gram_list = gram(patterns[i])
    gram_tags.extend([ind_tags[i]] * len(gram_list))
    grams.extend(gram_list)
# print(len(grams), grams)
# print(len(gram_tags), gram_tags)
# print(grams[26])
# print(gram_tags[26])

# numbering patterns
for i in range(len(grams)):
    return_num_list = []
    for num in tokenizer.texts_to_sequences(grams[i]):
        return_num_list.append(num[0])
    grams[i] = return_num_list
# print(len(grams), grams)
# print(grams[26])

# padding
grams = pad_sequences(grams, truncating='post', maxlen=50)
# print(grams[26])

grams = np.array(grams, dtype=float)
gram_tags = np.array(gram_tags, dtype=float)
print("grams\nshape:", grams.shape, "content:", grams)
print("gram_tags\nshape:", gram_tags.shape, "content:", gram_tags)

"""model training"""
# model = models.Sequential([
#     layers.Input(shape=[50]),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(num_classes, activation='softmax')
# ])
model = models.load_model("skipgram")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(grams, gram_tags, epochs=10000)
model.save("skipgram")