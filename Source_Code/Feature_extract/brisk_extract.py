import cv2 as cv
import numpy as np
import os

def brisk_extract(path_img, path_output):
    listdir  = os.listdir(path_img)
    count = 0
    for filename in listdir:
      pathfile = os.path.join(path_img, filename)
      path_outfile = os.path.join(path_output, filename)
      img = cv.imread(pathfile, 0)
      surf = cv.BRISK_create()
      kp, des = surf.detectAndCompute(img,None)
      np.savez(path_outfile, des)
      count += 1
      print(f"file {count} done!!")

class brisk_extract:
  def extract(self, img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    brisk = cv.BRISK_create()
    kp, des = brisk.detectAndCompute(img,None)
    return des
  def descriptorSize(self):
    brisk = cv.BRISK_create()      
    return brisk.descriptorSize()   

# path_img = "/media/lmtruong/01D557A8C9369F20/Work/JPEGImages"
# path_output = "/home/lmtruong/Documents/Work_Project/Data/Feature_extract/brisk_extract"
# brief_extract(path_img, path_output)
