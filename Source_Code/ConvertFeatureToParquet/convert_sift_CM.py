import convert_parquet
import numpy as np
import os

memory = "4g"
warehouse = "/home/lmtruong/Documents/Work_Project/Data/Feature_parquet"
path_input = "/home/lmtruong/Documents/Work_Project/Data/Feature_extract/sift_CM_extract"
name_path_output = "sift_CM_parquet"
convert_parquet.convert_parquet(memory, warehouse, path_input, name_path_output)