import os
import json
import cv2 as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class FeatureExtractor:
    def __init__(self, centers):
        self.feature_size = len(centers)
        self.knn = KNeighborsClassifier(n_neighbors=1, weights='distance')
        self.knn.fit(centers, range(self.feature_size))
        self.sift = cv.xfeatures2d.SIFT_create()
    
    def extract(self, gray):
        kp, des = self.sift.detectAndCompute(gray, None)
        pred = self.knn.predict(des)

        counter = np.zeros(self.feature_size, dtype=np.int32)
        for x in pred: counter[x] += 1

        # feature = np.linalg.norm(counter)
        return counter / sum(counter)#feature
