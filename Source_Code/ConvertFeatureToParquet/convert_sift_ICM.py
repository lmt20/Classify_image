import convert_parquet
import numpy as np
import os

memory = "4g"
warehouse = "/home/lmtruong/Documents/Work_Project/Data/Feature_parquet"
path_input = "/media/lmtruong/01D557A8C9369F20/Work/Feature_extract/sift_ICM_extract"
name_path_output = "sift_icm_parquet"
convert_parquet.convert_parquet(memory, warehouse, path_input, name_path_output)
