# from tensorflow import keras
# import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from matplotlib import pyplot
import random
# import tensorflow as tf
# import tensorflow.keras.layers
# import tensorflow.keras.models
# from tensorflow.keras.datasets import mnist
import numpy as np
import cv2


def firs_ocr():
    model = tf.keras.models.load_model("cnn_baseline")

    img = cv2.imread("test.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('og', img)
    edges = cv2.Canny(img, 175, 175)
    cv2.imshow('edges', edges)
    #
    cnts, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in cnts:
        contour = c.reshape(-1, 2)
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 15 and h > 15:
            rects.append([x, y, w, h])
            print(w, h)
        for coord in contour:
            cv2.circle(img, (coord[0], coord[1]), 1, (0, 0, 0))

    chars = []
    for r in rects:
        cv2.rectangle(edges, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (52, 249, 255), thickness=1)
        roi = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV)[1]

        (tH, tW) = roi.shape
        c_max, c_min = max(tH, tW), min(tH, tW)
        d = int((c_max-c_min)/2)
        if c_max == tW:
            roi = cv2.copyMakeBorder(roi, top=d+6, bottom=d+6, left=6, right=6, borderType=cv2.BORDER_CONSTANT, value=(0., 0., 0.))
        else:
            roi = cv2.copyMakeBorder(roi, top=6, bottom=6, left=d+6, right=d+6, borderType=cv2.BORDER_CONSTANT, value=(0., 0., 0.))
        roi = cv2.resize(roi, (28, 28))
        roi = np.reshape(roi, (28, 28, 1))
        chars.append(roi)

    cv2.imshow('rects', edges)
    print(len(chars))

    chars = np.array(chars)

    prediction = model.predict(chars)
    print(prediction.shape)
    letters = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(len(chars)):
        cv2.imshow('chars' + str(i), chars[i])
        print(letters[np.argmax(prediction[i])-1], i)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def word_splitter():
    img = cv2.imread('test.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_pics = False

    edges = cv2.Canny(img, 20, 20)
    if show_pics:
        cv2.imshow('edges', edges)
        cv2.imshow('og', img)

    contours, h_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in contours:
        contour = c.reshape(-1, 2)
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 15 and h > 15:
            rects.append([x, y, w, h])

    xs, ys = [], []
    for (x, y, w, h) in rects:
        xs.append(x)
        xs.append(x+w)
        ys.append(y)
        ys.append(y+h)

    top_left, bottom_right = (min(xs), min(ys)), (max(xs), max(ys))
    cv2.rectangle(edges, top_left, bottom_right, thickness=2, color=(52, 249, 255))
    word = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    word = cv2.resize(word, (word.shape[1], 50))
    word = cv2.threshold(word, 0, 255, cv2.THRESH_BINARY_INV)[1]
    print(word.shape)
    cv2.imshow('word', word)

    if show_pics:
        cv2.imshow('big_rect', edges)
        cv2.imshow('word', word)


    def split(im, f, stride):
        dy, dx = im.shape[0] - f[0], im.shape[1] - f[1]
        box = [1 + int(dy / stride[0]), 1 + int(dx / stride[1])]
        print(box)
        corner = [0, 0]
        pieces = []

        if box[0] < 1:
            box[0] = 1

        for y in range(box[0]):
            for x in range(box[1]):
                pieces.append(im[corner[0]:corner[0] + f[0], corner[1]:corner[1] + f[1]])
                corner[1] += stride[1]
            corner[0] += stride[0]
            corner[1] = 0

        return pieces


    p = split(word, [word.shape[0], 80], [1, 20])
    print(len(p))

    model = tf.keras.models.load_model("cnn_baseline")
    letters = 'abcdefghijklmnopqrstuvwxyz'

    for char in p:
        print(char.shape)
        inp = cv2.resize(char, (28, 28))
        inp = np.array([np.reshape(inp, (28, 28, 1))])
        pred = model.predict(inp)
        cv2.imshow(letters[np.argmax(pred) - 1], char)


def create_random(file_num):
    new_img = np.zeros((28, 28, 1), np.uint8)
    randy, randx = np.random.randint(29, size=6), np.random.randint(29, size=6)
    coords = [(randy[i:i+2], randx[i:i+2]) for i in range(len(randx)-1)]

    for (start, stop) in coords:
        cv2.line(new_img, start, stop, 255, 2)
    cv2.imwrite(r'C:\Users\Sabyasachi\PycharmProjects\TestGroundTwo\binary_data\random_img_' + file_num + '.png',
                new_img)

    if random.choice([0, 1, 1]) == 0:
        cv2.circle(img=new_img, center=(random.choice(randx), random.choice(randy)),
                   radius=random.randint(2, 18), color=255, thickness=2)

    # cv2.imshow('new_img', new_img)


def train_split():
    def load_data():
        scribbles_x = []
        for x in range(1, 70001):
            im = cv2.imread(r"binary_data\random_img_" + str(x) + ".png")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            scribbles_x.append(np.reshape(im, (28, 28, 1)))
        scribbles_y = np.array([0] * len(scribbles_x))

        (norm_train_x, y_), (norm_test_x, y__) = mnist.load_data()
        norm_x = np.concatenate((norm_train_x, norm_test_x), axis=0)
        norm_x = np.reshape(norm_x, (norm_x.shape[0], 28, 28, 1))
        norm_y = np.array([1] * len(norm_x))

        train_x = np.concatenate((scribbles_x, norm_x), axis=0)
        train_y = np.concatenate((scribbles_y, norm_y), axis=0)
        shuffle_idxs = np.random.permutation(len(train_y))
        train_x, train_y = train_x[shuffle_idxs], train_y[shuffle_idxs]

        test_idxs = np.random.randint(140000, size=20000)
        test_x, test_y = train_x[test_idxs], train_y[test_idxs]
        train_x, train_y = np.delete(train_x, test_idxs, axis=0), np.delete(train_y, test_idxs, axis=0)
        print(test_x.shape, train_y.shape, train_x.shape)

        for i in range(10):
            cv2.imshow("No" + str(i) + " " + str(train_y[i]), train_x[i])

    load_data()


# img = cv2.imread("test.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]
# # img = cv2.Canny(img, 100, 100)
# cv2.imshow('bgr', img)
#
# k = np.ones((2, 2), np.uint8)
# img = cv2.erode(img, k, cv2.BORDER_REFLECT)
# cv2.imshow('erode', img)
#
# cnts, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# new_img = np.zeros(img.shape)
#
# print(len(cnts))
# for c in range(0, len(cnts), 1):
#     contour = cnts[c].reshape(-1, 2)
#     for coord in contour:
#         cv2.circle(new_img, (coord[0], coord[1]), 1, (255, 255, 255))

# cv2.imshow('contours', new_img)

img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
k = np.ones([2, 2], np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
# img = cv2.erode(img, k, cv2.BORDER_REFLECT)
cv2.imshow('og', img)

edges = cv2.Canny(img, 0, 255)
cv2.imshow('edges', edges)

contours = np.zeros(edges.shape)
cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
rects = []
print(_)

for c in cnts:
    if c.shape[0] > 15:
        print(c.shape)
        temp = np.zeros(edges.shape)
        c = c.reshape(-1, 2)
        r = cv2.boundingRect(c)
        rects.append(r)
        for coord in c:
            cv2.circle(temp, (coord[0], coord[1]), 0, (255, 255, 255), thickness=-1)
        # cv2.imshow(str(c.shape[0]), temp)

for (x, y, w, h) in rects:
    cv2.rectangle(edges, [x, y], [x+w, y+h], (100, 100, 100), thickness=1)

cv2.imshow('rects', edges)

# cv2.imshow('contours', contours)

cv2.waitKey(0)
cv2.destroyAllWindows()