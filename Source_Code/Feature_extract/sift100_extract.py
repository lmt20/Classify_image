# import os
# import numpy as np
# import cv2 as cv


# def sift_extract(path_input_image, path_output_des):
#     listdir  = os.listdir(path_input_image)
#     count = 0
#     for filename in listdir:
#       pathfile = os.path.join(path_input_image, filename)
#       img = cv.imread(pathfile, 0)
#       sift = cv.xfeatures2d.SIFT_create(100)
#       kp, des = sift.detectAndCompute(img, None)
#       path_outfile = os.path.join(path_output_des, filename)
#       np.savez(path_outfile, des)
#       count += 1
#       print(f"file {count} done!!")


# path_img = "/media/lmtruong/01D557A8C9369F20/Work/JPEGImages"
# path_output = "/home/lmtruong/Documents/Work_Project/Data/Feature_extract/sift100_extract"
# sift_extract(path_img, path_output)
import os
import numpy as np
import cv2 as cv

class sift100_extract:

  def __init__(self):
    self.sift = cv.xfeatures2d.SIFT_create(100)

  def extract(self, img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, des = self.sift.detectAndCompute(gray, None)
    return des
  def descriptorSize(self):
      return 128

def save_extract(path_input_image, path_output_des):
    listdir  = os.listdir(path_input_image)
    count = 0
    for filename in listdir:
      pathfile = os.path.join(path_input_image, filename)
      img = cv.imread(pathfile, 0)
      sift = cv.xfeatures2d.SIFT_create(100)
      kp, des = sift.detectAndCompute(img, None)
      path_outfile = os.path.join(path_output_des, filename)
      np.savez(path_outfile, des)
      count += 1
      print(f"file {count} done!!")