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


def save_to_parquet(spark, files, batch_size, parquet_name):
    for i in range(0, len(files), batch_size):
        start = i
        end = i + batch_size

        print(f"Loading {start}..{end-1}")
        arrs = [np.load(x, allow_pickle = True) for x in files[start:end]]
        dataset = map(
            lambda x: (Vectors.dense(x), ),
            [x for arr in arrs if arr['arr_0'].ndim != 0  for x in arr["arr_0"]]
        )
        df = spark.createDataFrame(dataset, schema=["features"], samplingRatio=1)
        df.write.format('parquet').mode('append').saveAsTable('temporary')
        df.unpersist()

    # Compact files
    warehouse = spark.conf.get('spark.sql.warehouse.dir', 'spark-warehouse')
    tmp_parquet = os.path.join(warehouse, 'temporary')
    df = spark.read.parquet(tmp_parquet)
    df.write.format('parquet').mode('overwrite').saveAsTable(parquet_name)

    # Clean-up
    shutil.rmtree(tmp_parquet)  

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




def cal_centers(memory, warehouse, path_input, name_path_output, path_output_rsfile):
  spark = create_session(memory,warehouse)
  path = path_input
  listdir = os.listdir(path)
  listfile = [os.path.join(path, filename) for filename in listdir]
  save_to_parquet(spark, listfile, 100 , name_path_output)
  df = spark.read.parquet(os.path.join(warehouse,name_path_output))
  centers = cluster(df, 256, 100)
  np.save(path_output_rsfile, centers)
  print("Done")  

def cal_centers_after_have_parquet(memory, warehouse, name_path_output, path_output_rsfile, ncluser):
  spark = create_session(memory,warehouse)
  df = spark.read.parquet(os.path.join(warehouse,name_path_output))
  centers = cluster(df, ncluser, 100)
  np.save(path_output_rsfile, centers)
  print("Done")

# #first cal parquet and harrislaplace_CMI_128_centroid
# print("Begin")
# memory = "4g"
# warehouse = "/home/lmtruong/Documents/Work_Project/Project1/Kmean"
# path_input = "/media/lmtruong/01D557A8C9369F20/Work/FeatureExtract/harris-laplace_CMI"
# name_path_output = "harrislaplace_parquet"
# path_output_rsfile = "/home/lmtruong/Documents/Work_Project/Project1/Kmean/harrislaplace_CMI_128_centroid"
# cal_centers(memory, warehouse, path_input, name_path_output, path_output_rsfile)


#cal harrislaplace_CMI_256_centroid
print("Begin")
memory = "4g"
warehouse = "/home/lmtruong/Documents/Work_Project/Project1/Kmean"
name_path_output = "harrislaplace_parquet"
path_output_rsfile = "/home/lmtruong/Documents/Work_Project/Project1/Kmean/harrislaplace_CMI_256_centroid"
ncluster = 256
cal_centers_after_have_parquet(memory, warehouse, name_path_output, path_output_rsfile, ncluster)


path_output_rsfile = "/home/lmtruong/Documents/Work_Project/Project1/Kmean/harrislaplace_CMI_196_centroid"
ncluster = 196
cal_centers_after_have_parquet(memory, warehouse, name_path_output, path_output_rsfile, ncluster)
