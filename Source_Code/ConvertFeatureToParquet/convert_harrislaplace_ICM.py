import convert_parquet
import numpy as np
import os

memory = "4g"
warehouse = "/home/lmtruong/Documents/Work_Project/Data/Feature_parquet"
path_input = "/home/lmtruong/Documents/Work_Project/Data/Feature_extract/harrislaplace_ICM_extract"
name_path_output = "harrislaplace_ICM_parquet"
convert_parquet.convert_parquet(memory, warehouse, path_input, name_path_output)
