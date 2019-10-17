import os
import cv2 as cv
import numpy as np 
import random
import sys
import math
import Extract_BoW 
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from Feature_extract import brief_extract
from Feature_extract import brisk_extract
from Feature_extract import harrislaplace_CM_extract
from Feature_extract import harrislaplace_ICM_extract
from Feature_extract import sift_CM_extract
from Feature_extract import sift_extract
from Feature_extract import sift_ICM_extract
from Feature_extract import sift100_CM_extract
from Feature_extract import sift100_extract
from Feature_extract import sift100_ICM_extract
from Feature_extract import surf64_extract
from Feature_extract import surf128_extract

def prepare_data(dirpath, ratio):

    #extract file and label
    listdir = os.listdir(dirpath)
    listpathfile = []
    file_label = {}
    for dirname in listdir:
        listfile = os.listdir(os.path.join(dirpath,dirname))
        for filename in listfile:
            pathfile = os.path.join(dirpath, dirname, filename)
            listpathfile.append(pathfile)
            file_label[pathfile] = dirname

    #split train set and test set
    random.shuffle(listpathfile)
    size1 = int(len(listpathfile)*ratio)

    train_set = listpathfile[:size1]
    test_set = listpathfile[size1:]
    return (file_label, train_set, test_set)


def extract_encode_label(file_label, centroids, extractor):
    file_encode = {}
    extract_BoW = Extract_BoW.Extract_BoW(centroids, extractor)
    for pathfile in file_label:
        img = cv.imread(pathfile)
        encode = extract_BoW.extract(img)
        file_encode[pathfile] = encode
    return file_encode
    

def validate(dirpath, ratio, centroids, extractor):
    #split train and test set
    file_label, train_set, test_set = prepare_data(dirpath, ratio)
    #extract BoW for data
    time1 = time.time()
    file_encode = extract_encode_label(file_label, centroids, extractor)
    time2 = time.time()
    print("time1", time2-time1)
    # Training
    time1 = time.time()
    trainX = [file_encode[i] for i in train_set]
    trainY = [file_label[i] for i in train_set]
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    clf.fit(trainX, trainY)
    time2 = time.time()
    print("time_train", time2- time1)
    

    #Testing and validate
    right = 0
    testX = [file_encode[i] for i in test_set]
    testY = [file_label[i] for i in test_set]

    fig=plt.figure(figsize=(9,9))
    size = int(math.sqrt(len(testX)))
    
    for index, encode in enumerate(testX):    
        pred = clf.predict([encode])
        img  = cv.imread(test_set[index])
        sub = fig.add_subplot(size+1, size+1, index+1)
        sub.set_xticks([]) 
        sub.set_yticks([])
        if pred != testY[index]:
            sub.set_title(pred, color = "r")
        else:
            sub.set_title(pred)


        plt.imshow(img)
        if pred == testY[index]:
            right += 1

    accurate = right/len(testX)
    fig.suptitle(f"Rate: {right}/{len(testX)}~{accurate}", fontsize=16)
    
    return accurate

#extractor
brief_extractor = brief_extract.brief_extract()
brisk_extractor = brisk_extract.brisk_extract()
harrislaplace_CM_extractor = harrislaplace_CM_extract.harrislaplace_CM_extract()
harrislaplace_ICM_extractor = harrislaplace_ICM_extract.harrislaplace_ICM_extract()
sift_CM_extractor = sift_CM_extract.sift_CM_extract()
sift_extractor = sift_extract.sift_extract()
sift_ICM_extractor = sift_ICM_extract.sift_ICM_extract()
sift100_CM_extractor = sift100_CM_extract.sift100_CM_extract()
sift100_extractor = sift100_extract.sift100_extract()
sift100_ICM_extractor = sift100_ICM_extract.sift100_ICM_extract()
surf64_extractor = surf64_extract.surf64_extract()
surf128_extractor = surf128_extract.surf128_extract()

dict_extractor = {"brief":brief_extractor, "brisk":brisk_extractor,  \
     "harrislaplace_CM":harrislaplace_CM_extractor, "harrislaplace_ICM":harrislaplace_ICM_extractor,  \
         "sift":sift_extractor, "sift100":sift100_extractor,  \
         "sift_CM":sift_CM_extractor, "sift_ICM":sift_ICM_extractor,  \
             "sift100_CM":sift100_CM_extractor, "sift100_ICM":sift100_ICM_extractor,  \
                 "surf64":surf64_extractor, "surf128":surf128_extractor
     }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path_centroids')
    parser.add_argument('extractor')
    options = parser.parse_args()
    dirpath = "/home/lmtruong/Pictures/data"
    ratio = 0.8
    centroids = np.load(options.path_centroids)
    extractor = dict_extractor[options.extractor]

    accuracy = validate(dirpath, ratio, centroids, extractor)
    plt.show()

    # sum = 0
    # for i in range(50):
    #     accuracy = validate(dirpath, ratio, centroids, extractor)
    #     sum += accuracy
    #     print(accuracy)
    # print("Average precise:", sum/50)

#use: python3 train_test.py /home/lmtruong/Documents/Work_Project/Data/Centroid_extract/sift100_centroids256.npy sift100