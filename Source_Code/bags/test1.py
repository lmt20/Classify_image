import os 
import numpy as np
from pyspark.sql import SparkSession


# path = "/home/lmtruong/Documents/Work_Project/Test/npzFile"
# path = "/home/lmtruong/Documents/Work_Project/Project1/brift_extract"
# listdir = os.listdir(path)
# listfile = [os.path.join(path, filename) for filename in listdir]
# listfile = listfile[315:320]
# load_file = [np.load(filename, allow_pickle = True) for filename in listfile]


# arr_1d = [x for arr in load_file if arr['arr_0'].ndim != 0 for x in arr["arr_0"]]
# print(arr_1d)

# arr_rs = []
# arr_test = load_file[0]["arr_0"]
# arr_2d = [file['arr_0'] for file in load_file if file['arr_0'] is not None]
# arr_1d = []
# for file_npz in load_file:
#     arr_2d = file_npz['arr_0']
#     if arr_2d.ndim != 0:
#         for arr in arr_2d:
#             arr_1d.append(arr)
    # if arr_2d['arr0'] is not None:
        # for arr1d in arr_2d:
        #     arr_1d.append(arr1d)
# print(arr_1d)
# print(len(arr_1d))
# print(len(arr_1d[0]))

# arr_1d = [x for arr in arr_2d for x in arr]
# print(arr_1d)


# for file in load_file:
#     if file['arr_0'] is not None:
#         print(file['arr_0'])
#         # for x in file['arr_0']:
#         #     arr_rs.append(x)
# print(arr_rs)

# arr = [x['arr_0'] for x in load_file if x['arr_0'] is not None] 
# print([len(x) for x in arr])
# arr_npz = np.load(listfile[2], allow_pickle = True)
# print(arr_npz['arr_0'])
path = "/home/lmtruong/Documents/Work_Project/Test/npzFile"
# path = "/home/lmtruong/Documents/Work_Project/Project1/brift_extract"
listdir = os.listdir(path)
listfile = [os.path.join(path, filename) for filename in listdir]
length = 0
for file in listfile:
    arr = np.load(file)
    arr_rs = arr['arr_0']
    # print(len(arr_rs))
    length += len(arr_rs)
print("length", length)

# arrs = [np.load(i ,allow_pickle = True) for i in listfile]
# # list_rs = []
# # for arr in arrs:
# #     if arr["arr_0"] != None:
# #         for x in arr:
# #             list_rs.append(x)
# # print(list_rs)

# # list_rs = [x for arr in arrs if arr["arr_0"] != None for x in arr["arr_0"]]
# # print(list_rs)

# listarr = [i for i in arrs]
# print(listarr)
# print(type(listarr))
# arr_rs = arrs['arr_0']
# print(arr_rs)

# # for file in listfile:
# #     arr = np.load(file)
# #     arr_rs = arr['arr_0']
# #     print(arr_rs[0])
# #     break
spark = SparkSession.builder.appName("Myapp").getOrCreate()
path = "/home/lmtruong/Documents/Work_Project/Project1/bags/test/testmd"
listdir = os.listdir(path)
print(len(listdir))
# listfile = [os.path.join(path, filename) for filename in listdir if filename.endswith(".parquet")]
df = spark.read.parquet(path)
result = df.collect()   
print(len(result))
# print(result[12411])
