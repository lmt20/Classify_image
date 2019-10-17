import os
import shutil
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans


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

        print(f"Loading {start}..{end}")
        arrs = [np.load(x) for x in files[start:end]]
        dataset = map(
            lambda x: (Vectors.dense(x), ),
            [x for arr in arrs for x in arr]
        )

        df = spark.createDataFrame(dataset, schema=["features"], samplingRatio=1)
        df.write.format('parquet').mode('append').saveAsTable('_temporary')
        df.unpersist()

    # Compact files
    warehouse = spark.conf.get('spark.sql.warehouse.dir', 'spark-warehouse')
    tmp_parquet = os.path.join(warehouse, '_temporary')
    
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--memory', required=True)
    parser.add_argument('--k', required=True, type=int)
    parser.add_argument('--iter', required=True, type=int)
    options = parser.parse_args()
    print(options)

    spark = create_session(options.memory, 'warehouse')
    df = spark.read.parquet(options.input)
    centers = cluster(df, k=options.k, maxIter=options.iter)
    np.save(options.output, centers)
