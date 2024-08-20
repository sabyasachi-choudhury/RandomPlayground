from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import tensorflow as tf

(train_x, train_y), (test_x, test_y) = mnist.load_data()
test_x = tf.reshape(test_x, (test_x.shape[0], 28, 28, 1))

# loading model
model = load_model("self_convolutional")
predictor = tf.keras.models.Sequential([
    model,
    tf.keras.layers.Softmax()
])

res = model.evaluate(test_x, test_y, verbose=2)
print(res)