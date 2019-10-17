import os
import argparse
import cv2 as cv
import numpy as np
from random import shuffle, Random
from sklearn.neighbors import KNeighborsClassifier
from extract import FeatureExtractor


def load_data(directory, ratio, random_state):
    labels = []
    encodings = []

    data = {}
    for item in os.listdir(directory):
        label = item[:item.find('_')]
        arr = data.get(label, [])
        arr.append(os.path.join(directory, item))
        data[label] = arr

    a = []
    b = []
    r = Random(random_state)
    for label in data:
        arr = data[label]
        shuffle(arr, r.random)
               
        mid = int(round(len(arr) * ratio))
        for x in arr[mid:]: a.append((x, label))
        for x in arr[:mid]: b.append((x, label))
    
    return a, b

def prepare(dataset, extract_fun):
    encodes = []
    labels = []
    for filename, label in dataset:
        img = cv.imread(filename)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        encodes.append(extract_fun(gray))
        labels.append(label)
    
    return encodes, labels


parser = argparse.ArgumentParser()
parser.add_argument("--ratio", type=float, default=.3)
parser.add_argument("directory")
options = parser.parse_args()


centers = np.load('VOC/centers256.npy')
extractor = FeatureExtractor(centers)

def validate(directory, ratio, random_state):
    # Split train and test set
    train, test = load_data(directory, ratio, random_state)

    # Training
    trainX, trainY = prepare(train, extractor.extract)
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf.fit(trainX, trainY)

    # Validate on test set
    wrong = 0
    testX, testY = prepare(test, extractor.extract)
    for i in range(len(testX)):
        encode = testX[i]
        label = testY[i]
        filename, _ = test[i]

        pred, = clf.predict([encode])
        wrong += pred != label
        # print('+', filename, pred)

    # print('Total:', len(testX))
    # print('Wrong:', wrong)
    # print('Accuracy:', 1 - wrong / len(testX))
    return 1 - wrong / len(testX)

accuracy = [validate(options.directory, options.ratio, i) for i in range(10)]
print("Avg. accuracy:", sum(accuracy) / len(accuracy))
