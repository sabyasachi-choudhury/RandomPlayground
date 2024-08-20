# Imports
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAvgPool1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

patterns = []
ind_tags = []
labels = []
with open("testing.json") as file:
    data = json.load(file)

for intent_set in data['intents']:
    for pattern in intent_set['patterns']:
        patterns.append(pattern)
        ind_tags.append(intent_set['tag'])
    if intent_set['tag'] not in labels:
        labels.append(intent_set['tag'])

total_classes = len(labels)
print("og_patterns", patterns)

# print(ind_tags)
lbl_encoder = LabelEncoder()
lbl_encoder.fit(ind_tags)
ind_tags = lbl_encoder.transform(ind_tags)
print("encoded_ind_tags: ", ind_tags)

vocabulary = 1000
embed_dim = 16
oov_token = '<OOV>'
max_len = 20

tokenizer = Tokenizer(num_words=vocabulary, oov_token=oov_token)
tokenizer.fit_on_texts(patterns)
print("fit_on_textx: ", patterns)
word_index = tokenizer.word_index
print(word_index)
sequences = tokenizer.texts_to_sequences(patterns)
print("text to seq", sequences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
print(padded_sequences)

model = Sequential()
model.add(Embedding(vocabulary, embed_dim, input_length=max_len))
model.add(GlobalAvgPool1D())
model.add(Dense(16, activation='relu', input_shape=(20, )))
model.add(Dense(16, activation='relu'))
model.add(Dense(total_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
# model.fit(padded_sequences, np.array(ind_tags), epochs=800)
# model.save("experiment")

# to save the fitted tokenizer
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# #
# # # to save the fitted label encoder
# with open('label_encoder.pickle', 'wb') as ecn_file:
#     pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)


# model = tf.keras.models.load_model("experiment")
# # load tokenizer object
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
#
# # load label encoder object
# with open('label_encoder.pickle', 'rb') as enc:
#     lbl_encoder = pickle.load(enc)
#
# with open("testing.json") as file:
#     data = json.load(file)
#
#
# def predict(sentence):
#     result = model.predict(pad_sequences(tokenizer.texts_to_sequences([sentence]),
#                                          truncating='post', maxlen=20))
#     tag = lbl_encoder.inverse_transform([np.argmax(result)])
#     print(tag)
#     for i in data["intents"]:
#         if i["tag"] == tag:
#             print(random.choice(i["responses"]))
#
#
# predict("hi")
# predict("goodbye")