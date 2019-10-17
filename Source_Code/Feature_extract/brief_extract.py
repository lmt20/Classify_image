import cv2 as cv
import numpy as np
import os

def brief_extract(path_img, path_output):
    listdir  = os.listdir(path_img)
    count = 0
    for filename in listdir:
      pathfile = os.path.join(path_img, filename)
      path_outfile = os.path.join(path_output, filename)
      img = cv.imread(pathfile, 0)
      star = cv.xfeatures2d.StarDetector_create()
      brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
      kp = star.detect(img,None)
      kp, des = brief.compute(img, kp)
      np.savez(path_outfile, des)
      count += 1
      print(f"file {count} done!!")

class brief_extract:
  def extract(self, img):
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      star = cv.xfeatures2d.StarDetector_create()
      brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
      kp = star.detect(img,None)
      kp, des = brief.compute(img, kp)    
      return des
  def descriptorSize(self):
      star = cv.xfeatures2d.StarDetector_create()
      brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
      return brief.descriptorSize()
# path_img = "/media/lmtruong/01D557A8C9369F20/Work/JPEGImages"
# path_output = "/home/lmtruong/Documents/Work_Project/Data/Feature_extract/brief_extract"
# brief_extract(path_img, path_output)


