import numpy as np
import os

# path ="/media/lmtruong/01D557A8C9369F20/Work/Feature_extract/brisk_extract"
# listdir = os.listdir(path)
# list_pathfile = [os.path.join(path, filename) for filename in listdir]
# count = 0
# for pathfile in list_pathfile:
#     arr_npz = np.load(pathfile)
#     arr_np = arr_npz['arr_0']
#     print(arr_np.shape)

path = "/media/lmtruong/01D557A8C9369F20/Work/Feature_extract/sift100_ICM_extract1/2007_000272.jpg.npz"    
arr_npz = np.load(path)
arr_np = arr_npz['arr_0']
print(len(arr_np))
# count = 0
# for i in arr_np:
#     print(i)
#     if np.isnan(i[0]):
#         print(i, count)
#         break
#     count+=1
path = "/media/lmtruong/01D557A8C9369F20/Work/Feature_extract/test_extract/2007_000272.jpg.npz"    
arr_npz = np.load(path)
arr_np = arr_npz['arr_0']
print(len(arr_np))

# path ="/media/lmtruong/01D557A8C9369F20/Work/Feature_extract/test_extract"
# listdir = os.listdir(path)
# list_pathfile = [os.path.join(path, filename) for filename in listdir]
# count = 0
# for pathfile in list_pathfile:
#     arr_npz = np.load(pathfile)
#     arr_np = arr_npz['arr_0']
#     # shape = arr_np.shape
#     # print(shape)
#     # if arr_np.ndim == 2:
#     #     for i in arr_np:
#     #         print(i)
#     #     if shape[1] != 24:
#     #         print(count)
#     #         print(arr_np)
#     #         break
#     # count+=1
#     for i in arr_np:
#         print(i)
#     # print(arr_np)
#     print(arr_np.shape)
    
