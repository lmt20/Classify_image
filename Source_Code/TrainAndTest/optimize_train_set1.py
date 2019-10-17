import os
import cv2 as cv
import numpy as np 
import random
import sys
import math
import Extract_BoW 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

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
from Feature_extract import sift50_extract


def extract_file_label(dirpath):
    #extract file and label
    listdir = os.listdir(dirpath)
    # listpathfile = []
    file_label = {}
    for dirname in listdir:
        listfile = os.listdir(os.path.join(dirpath,dirname))
        for filename in listfile:
            pathfile = os.path.join(dirpath, dirname, filename)
            # listpathfile.append(pathfile)
            file_label[pathfile] = dirname
    return file_label

def extract_encode_label(file_label, centroids, extractor):
    file_encode = {}
    extract_BoW = Extract_BoW.Extract_BoW(centroids, extractor)
    for pathfile in file_label:
        img = cv.imread(pathfile)
        encode = extract_BoW.extract(img)
        file_encode[pathfile] = encode
    return file_encode

def split_data(file_label, ratio):

    #split train set and test set
    listpathfile = [key for key in file_label]
    random.shuffle(listpathfile)
    size1 = int(len(listpathfile)*ratio)
    train_set = listpathfile[:size1]
    test_set = listpathfile[size1:]
    return (train_set, test_set)

def split_train_validate_set(data, ratio):
    size = int(len(data)*(1-ratio))
    num_batch = int(len(data)/size)
    train_set = []
    validity_set = []
    for i in range(num_batch):
        validity = data[size*i:size*(i+1)]
        train = data[0:size*i]+data[size*(i+1):]
        train_set.append(train)
        validity_set.append(validity)
    return train_set, validity_set


def cross_validation(data_trainset, ratio, file_label, file_encode):
    #split train_set --> train_set||validate_set
    train_set, test_set = split_train_validate_set(data_trainset, ratio)
    #test average score for n_neighbors: k:1-->9
    anchor = 1
    best_score = 0
    for k in range(1,10):
        sum = 0
        for i in range(len(train_set)):
            sum += cal_score(file_label, file_encode, train_set[i], test_set[i], k)
        avg_score = sum/len(train_set)
        if best_score < avg_score:
            best_score = avg_score
            anchor = k
    #return anchor: best number n_neighbors
    return anchor
def cal_score(file_label, file_encode, train_set, test_set, k):
    # Training
    trainX = [file_encode[i] for i in train_set]
    trainY = [file_label[i] for i in train_set]
    clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
    clf.fit(trainX, trainY)
    #Testing and validate
    right = 0
    testX = [file_encode[i] for i in test_set]
    testY = [file_label[i] for i in test_set]
    for index, encode in enumerate(testX):    
        pred = clf.predict([encode])
        if pred == testY[index]:
            right += 1
    accurate = right/len(testX)
    return accurate

def cal_score_optimize(file_label, file_encode1, file_encode2, train_set, test_set, k):
    # Training
    trainX = [file_encode1[i] for i in train_set]
    trainY = [file_label[i] for i in train_set]
    testX = [file_encode1[i] for i in test_set]
    testY = [file_label[i] for i in test_set]

    #find 10 neighbors nearest each test image
    num_neighbor = 200 if k<200 else k
    neigh = NearestNeighbors(n_neighbors=num_neighbor)
    neigh.fit(trainX)
    arr_distance, arr_index = neigh.kneighbors(testX)

    #find nearest image using other algorithm in 10 image 
    right  = 0
    for i in range(len(test_set)):
        testX1 = file_encode2[test_set[i]]        
        name_10image = [train_set[j] for j in arr_index[i]]
        _10_labels = [file_label[i] for i in name_10image]
        print(f"Label: {_10_labels} \nTrueLabel: {testY[i]}" )
        trainX1 = [file_encode2[i] for i in name_10image]
        trainY1 = [file_label[i] for i in name_10image]
        neigh1 = NearestNeighbors(1)
        neigh1.fit(trainX1)
        arr_distance1, arr_index1 = neigh1.kneighbors([testX1])
        index = arr_index1[0][0]
        print(index, trainY1[index])
        #check prediction label
        true_label = testY[i]
        predict_label = trainY1[index]
        right += true_label==predict_label
    accurate = right/len(test_set)
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
sift50_extractor = sift50_extract.sift50_extract()

dict_extractor = {"brief":brief_extractor, "brisk":brisk_extractor,  \
     "harrislaplace_CM":harrislaplace_CM_extractor, "harrislaplace_ICM":harrislaplace_ICM_extractor,  \
         "sift":sift_extractor, "sift100":sift100_extractor,  \
         "sift_CM":sift_CM_extractor, "sift_ICM":sift_ICM_extractor,  \
             "sift100_CM":sift100_CM_extractor, "sift100_ICM":sift100_ICM_extractor,  \
                 "surf64":surf64_extractor, "surf128":surf128_extractor, \
                     "sift50":sift50_extractor
     }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path_centroids1')
    parser.add_argument('path_centroids2')
    parser.add_argument('extractor1')
    parser.add_argument('extractor2')
    options = parser.parse_args()

    dirpath = "/home/lmtruong/Pictures/data"
    # dirpath = "/home/lmtruong/Pictures/test_image"
    ratio = 0.8
    centroids1 = np.load(options.path_centroids1)
    centroids2 = np.load(options.path_centroids2)
    extractor1 = dict_extractor[options.extractor1]
    extractor2 = dict_extractor[options.extractor2]

    file_label = extract_file_label(dirpath)
    file_encode1 = extract_encode_label(file_label, centroids1, extractor1)
    print("Extractor 1 done!")
    file_encode2 = extract_encode_label(file_label, centroids2, extractor2)
    print("Extractor 2 done!")

    train_set, test_set = split_data(file_label, ratio)
    k_neighbor = cross_validation(train_set, ratio, file_label, file_encode1)
    # score = cal_score(file_label, file_encode, train_set, test_set, k_neighbor)
    # print(k_neighbor, score)    

    sum = 0
    for i in range(50):
        train_set, test_set = split_data(file_label, ratio)   
        # k_neighbor = cross_validation(train_set, ratio, file_label, file_encode1)
        score = cal_score_optimize(file_label, file_encode1, file_encode2, train_set, test_set, k_neighbor)
        # print(score)    
        sum += score

    #write to logfile   
    # f = open("/home/lmtruong/Documents/Work_Project/Data/Log_files/file_log_score.txt", "a")
    # message = f"Type extract: {options.extractor}  \
    #      \nN_cluster: {len(centroids) } \
    #      \nBest n_neighbor: {k_neighbor} \
    #      \nScore average: {sum/50}"
    # f.write(message)

    print("Type extract:", options.extractor1, "\ncluster:", len(centroids1))
    print("Score average:", sum/50)


"""python3 optimize_train_set1.py /home/lmtruong/Documents/Work_Project/Data/Centroid_extract/harrislaplace_icm_centroids128.npy /home/lmtruong/Documents/Work_Project/Data/Centroid_extract/sift100_centroids128.npy harrislaplace_ICM sift100 """