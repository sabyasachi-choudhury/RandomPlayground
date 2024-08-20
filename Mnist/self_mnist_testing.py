import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_img, test_lbl) = mnist.load_data()

model = tf.keras.models.load_model("self_mnist")
probability_model = tf.keras.models.Sequential([
    model,
    tf.keras.layers.Softmax()
])

test_img = tf.reshape(test_img, (test_img.shape[0], -1))
test_img = np.array(test_img, dtype=float)
test_img /= 255.0

# print(test_lbl[10])
# print(probability_model.predict(np.array([test_img[10]])))
# print(test_lbl)
# print(probability_model.predict(test_img))
prediction = probability_model.predict(test_img)
acc = []
for x in range(len(prediction)):
    if np.argmax(prediction[x]) == test_lbl[x]:
        acc.append(1)
    else:
        acc.append(0)

print(sum(acc)/len(acc))


"""YAAAASSSYDFIYEWRGTP3URTGVIEW5TQ3IUTVWQUT G4;I UTG REIUG W;RIUG WR;IUGWR TGRI;UGT"""