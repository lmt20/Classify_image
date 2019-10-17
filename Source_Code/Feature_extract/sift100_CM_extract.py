import cv2 as cv
import numpy as np
import os

    
def cal_moment(img):

  height, width = img.shape[0:2]
  #cal matrix x^p*y^q in three case (p,q) = [(0,0), (0,1), (1,0)]
  arr_xp_yq = [np.ones((height, width))]
  xp_yq = np.zeros((height, width))
  for i in range(height):
    for j in range(width):
      xp_yq[i][j] = j/width
  arr_xp_yq.append(xp_yq)
  xp_yq = np.zeros((height, width))
  for i in range(height):
    for j in range(width):
      xp_yq[i][j] = i/height
  arr_xp_yq.append(xp_yq)
  #cal matrix R^a * R^b * R^c in 10 case (a,b,c)
  arr_abc = [[0,0,0],[0,0,1],[0,1,0],[1,0,0],[1,1,0],[1,0,1],[0,1,1],[2,0,0],[0,2,0],[0,0,2]]
  Mat_B = img[:,:,0]/256
  Mat_G = img[: ,:,1]/256
  Mat_R = img[:,:,2]/256
  arr_MatB = [np.ones(Mat_B.shape), Mat_B, Mat_B*Mat_B]
  arr_MatG = [np.ones(Mat_B.shape), Mat_B, Mat_B*Mat_B]
  arr_MatR = [np.ones(Mat_B.shape), Mat_B, Mat_B*Mat_B]
  arr_chanel = []
  for i in range(10):
    arr_chanel.append(arr_MatB[arr_abc[i][0]]*arr_MatG[arr_abc[i][1]]*arr_MatR[arr_abc[i][2]]) 
#   cal 30 color moments
  arr_moments = []
  for i in range(3):
    for j in range(10):
      mat_agregation = arr_xp_yq[i] * arr_chanel[j]
      moment = np.sum(mat_agregation)
      arr_moments.append(moment)
  return arr_moments

def cal_siftCM_totalimage(img):
  gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  sift = cv.xfeatures2d.SIFT_create(100)
  kp, des = sift.detectAndCompute(gray_img, None)
  rs_moments =  [] 
  for i in range(len(kp)):
    x = int(kp[i].pt[0])
    y = int(kp[i].pt[1])
    radius = int((kp[i].size+1)/2)
    sub_img = img[y-radius:y+radius, x-radius:x+radius]
    rs_moments.append(cal_moment(sub_img))
  return np.array(rs_moments)
  
#SIFT-CM extract
def sift_CM_extract(path_img, path_output):
  print("begin!")
  list_dir = os.listdir(path_img)
  count = 0
  for filename in list_dir:
    pathfile = os.path.join(path_img, filename)
    outfile = os.path.join(path_output, filename)
    img = cv.imread(pathfile)
    feture_extract = cal_siftCM_totalimage(img)
    np.savez(outfile, feture_extract)
    count +=1
    print(f"extract feature file {count} done!!")

class sift100_CM_extract:
  def __init__(self):
    self.sift = cv.xfeatures2d.SIFT_create(100)
  def extract(self, img):
    return cal_siftCM_totalimage(img)
  def descriptorSize(self):
      return 30

# path_img = "/media/lmtruong/01D557A8C9369F20/Work/JPEGImages"
# path_output = "/home/lmtruong/Documents/Work_Project/Data/Feature_extract/sift_CM_extract"
# sift_CM_extract(path_img, path_output)