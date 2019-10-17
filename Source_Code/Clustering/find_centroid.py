import os
import shutil
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans


os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"

def create_session(memory, warehouse):
    return SparkSession.builder.appName("My app")       \
        .config('spark.driver.memory', memory)          \
        .config('spark.sql.warehouse.dir', warehouse)   \
        .config('spark.rdd.compress', True)            \
        .getOrCreate()

def cluster(dataframe, k, maxIter=50):
    clf = KMeans(k=k, maxIter=maxIter)
    model = clf.fit(dataframe)
    print('=================================')
    print('Cost:', model.summary.trainingCost)
    # print('Cluster:', model.summary.cluster)
    # print('Cluster centers:', model.clusterCenters())
    print('Cluster size:', model.summary.clusterSizes)
    print('Iter:', model.summary.numIter)
    # print('Predictions:', model.summary.predictions)
    print('k:', model.summary.k)
    print('featuresCol:', model.summary.featuresCol)
    print('predictionCol:', model.summary.predictionCol)
    #print(dir(model.summary))
    # print('=================================')
    # print('Saving centers...')
    # np.save('tmp/centers' + str(options.k), model.clusterCenters())
    # print('Saving predictions...')
    # model.summary.predictions.write.format('parquet').saveAsTable('predictions' + str(options.k))
    return model.clusterCenters()


def cal_centers(memory, warehouse, name_path_parquet, path_centroidfile, ncluster):
  spark = create_session(memory,warehouse)
  df = spark.read.parquet(os.path.join(warehouse,name_path_parquet))
  print("Read done! \n Begin clustering")
  centers = cluster(df, ncluster, 100)
  np.save(path_centroidfile, centers)
  print("Done")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('memory')
    parser.add_argument('warehouse')
    parser.add_argument('name_path_parquet')
    parser.add_argument('path_centroidfile')
    parser.add_argument('ncluster', type=int)
    options = parser.parse_args()
    print(options)
    cal_centers(options.memory, options.warehouse, options.name_path_parquet, options.path_centroidfile, options.ncluster)

 
# memory = "4g"
# warehouse = "/home/lmtruong/Documents/Work_Project/Source_Code/bags/test"
# name_path_parquet = "testmd"
# path_centroidfile = "/home/lmtruong/Documents/Work_Project/Source_Code/bags/test/cetroid_test1"
# ncluster = 128
# cal_centers(memory, warehouse, name_path_parquet, path_centroidfile, ncluster)

#python3 find_centroid.py 4g /home/lmtruong/Documents/Work_Project/Data/Feature_parquet  brief_parquet /home/lmtruong/Documents/Work_Project/Data/Centroid_extract/brief_centroids64 64