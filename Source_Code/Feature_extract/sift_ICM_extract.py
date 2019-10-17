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
  arr_abc = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[2,0,0],[0,2,0],[0,0,2],[1,1,0],[1,0,1],[0,1,1]]
  Mat_B = img[:,:,0]/256
  Mat_G = img[: ,:,1]/256
  Mat_R = img[:,:,2]/256
  arr_MatB = [np.ones(Mat_B.shape), Mat_B, Mat_B*Mat_B]
  arr_MatG = [np.ones(Mat_B.shape), Mat_B, Mat_B*Mat_B]
  arr_MatR = [np.ones(Mat_B.shape), Mat_B, Mat_B*Mat_B]
  arr_chanel = []
  for i in range(10):
    arr_chanel.append(arr_MatB[arr_abc[i][0]]*arr_MatG[arr_abc[i][1]]*arr_MatR[arr_abc[i][2]]) 
#   cal 30 moments
  arr_moments = []
  for i in range(3):
    for j in range(10):
      mat_agregation = arr_xp_yq[i] * arr_chanel[j]
      moment = np.sum(mat_agregation)
      arr_moments.append(moment)
  return arr_moments

def cal_invariant_colormoments(img):
  arr_moments = cal_moment(img)
  M00 = arr_moments[0:10]
  M01 = arr_moments[10:20]
  M10 = arr_moments[20:]
  arr_invariant_moment = []
  #cal S02
  for i in range(1,4,1):
    S02 = (M00[3+i]*M00[0])/(M00[i]*M00[i])
    arr_invariant_moment.append(S02)
  #cal D02
  for i in range(1,4,1):
    D02 = (M00[6+i]*M00[0])/(M00[i]*M00[i%3+1])
    arr_invariant_moment.append(D02)
  #cal S12
  for i in range(1,4,1):
    index = {"0":0, "1":i, "2":3+i, "11":6+i, "10":i, "01":i%3+1, "20":3+i, "02":4+i%3}
    S12 = (M10[index["2"]]*M01[index["0"]]*M00[index["1"]] \
          +M10[index["1"]]*M01[index["2"]]*M00[index["0"]] \
          +M10[index["0"]]*M01[index["1"]]*M00[index["2"]] \
          -M10[index["2"]]*M01[index["1"]]*M00[index["0"]] \
          -M10[index["1"]]*M01[index["0"]]*M00[index["2"]] \
          -M10[index["0"]]*M01[index["2"]]*M00[index["1"]] \
           ) / (M00[index["2"]]*M00[index["1"]]*M00[index["0"]])
    arr_invariant_moment.append(S12)
  #cal D11
  for i in range(1,4,1):
    index = {"0":0, "1":i, "2":3+i, "11":6+i, "10":i, "01":i%3+1, "20":3+i, "02":4+i%3}
    D11 = (M10[index["10"]]*M01[index["01"]]*M00[index["0"]] \
          + M10[index["01"]]*M01[index["0"]]*M00[index["10"]] \
          + M10[index["0"]]*M01[index["10"]]*M00[index["01"]] \
          - M10[index["10"]]*M01[index["0"]]*M00[index["01"]] \
          - M10[index["01"]]*M01[index["10"]]*M00[index["0"]] \
          - M10[index["0"]]*M01[index["01"]]*M00[index["10"]]
          ) / (M00[index["2"]]*M00[index["1"]]*M00[index["0"]])
    arr_invariant_moment.append(D11)
  #cal D12_1
  for i in range(1,4,1):
    index = {"0":0, "1":i, "2":3+i, "11":6+i, "10":i, "01":i%3+1, "20":3+i, "02":4+i%3}
    D12_1 = (M10[index["11"]]*M01[index["0"]]*M00[index["10"]] \
          + M10[index["10"]]*M01[index["11"]]*M00[index["0"]] \
          + M10[index["0"]]*M01[index["10"]]*M00[index["11"]] \
          - M10[index["11"]]*M01[index["10"]]*M00[index["0"]] \
          - M10[index["10"]]*M01[index["0"]]*M00[index["11"]] \
          - M10[index["0"]]*M01[index["11"]]*M00[index["10"]]
          ) / (M00[index["11"]]*M00[index["10"]]*M00[index["0"]])
    arr_invariant_moment.append(D12_1)
  #cal D12_2
  for i in range(1,4,1):
    index = {"0":0, "1":i, "2":3+i, "11":6+i, "10":i, "01":i%3+1, "20":3+i, "02":4+i%3}
    D12_2 = (M10[index["11"]]*M01[index["0"]]*M00[index["01"]] \
          + M10[index["01"]]*M01[index["11"]]*M00[index["0"]] \
          + M10[index["0"]]*M01[index["01"]]*M00[index["11"]] \
          - M10[index["11"]]*M01[index["01"]]*M00[index["0"]] \
          - M10[index["01"]]*M01[index["0"]]*M00[index["11"]] \
          - M10[index["0"]]*M01[index["11"]]*M00[index["01"]]
          ) / (M00[index["11"]]*M00[index["01"]]*M00[index["0"]])
    arr_invariant_moment.append(D12_2)
  #cal D12_3
  for i in range(1,4,1):
    index = {"0":0, "1":i, "2":3+i, "11":6+i, "10":i, "01":i%3+1, "20":3+i, "02":4+i%3}
    D12_3 = (M10[index["02"]]*M01[index["0"]]*M00[index["10"]] \
          + M10[index["10"]]*M01[index["02"]]*M00[index["0"]] \
          + M10[index["0"]]*M01[index["10"]]*M00[index["02"]] \
          - M10[index["02"]]*M01[index["10"]]*M00[index["0"]] \
          - M10[index["10"]]*M01[index["0"]]*M00[index["02"]] \
          - M10[index["0"]]*M01[index["02"]]*M00[index["10"]]
          ) / (M00[index["02"]]*M00[index["10"]]*M00[index["0"]])
    arr_invariant_moment.append(D12_3)
  #cal D12_4
  for i in range(1,4,1):
    index = {"0":0, "1":i, "2":3+i, "11":6+i, "10":i, "01":i%3+1, "20":3+i, "02":4+i%3}
    D12_4 = (M10[index["20"]]*M01[index["01"]]*M00[index["0"]] \
          + M10[index["01"]]*M01[index["0"]]*M00[index["20"]] \
          + M10[index["0"]]*M01[index["20"]]*M00[index["01"]] \
          - M10[index["20"]]*M01[index["0"]]*M00[index["01"]] \
          - M10[index["01"]]*M01[index["20"]]*M00[index["0"]] \
          - M10[index["0"]]*M01[index["01"]]*M00[index["20"]]
          ) / (M00[index["20"]]*M00[index["01"]]*M00[index["0"]])
    arr_invariant_moment.append(D12_4)
  return arr_invariant_moment

def cal_sift_CMI_totalimage(img):
  gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  sift = cv.xfeatures2d.SIFT_create()
  kp, des = sift.detectAndCompute(gray_img, None)
  # gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  # harr_lap = cv.xfeatures2d.HarrisLaplaceFeatureDetector_create()
  # kp = harr_lap.detect(gray_img)
  rs_moments =  [] 
  for i in range(len(kp)):
    x = int(kp[i].pt[0])
    y = int(kp[i].pt[1])
    radius = int((kp[i].size)/2)
    if radius != 0:
      sub_img = img[y-radius:y+radius, x-radius:x+radius]
      arr_sub = cal_invariant_colormoments(sub_img)
      if not any(np.isnan(arr_sub)):
        rs_moments.append(arr_sub)
  return np.array(rs_moments)

# feature extraction
def sift_ICM_extract(path_img, path_output):
  print("begin!")
  list_dir = os.listdir(path_img)
  count = 0
  for filename in list_dir:
    pathfile = os.path.join(path_img, filename)
    outfile = os.path.join(path_output, filename)
    img = cv.imread(pathfile)
    feture_extract = cal_sift_CMI_totalimage(img)
    np.savez(outfile, feture_extract)
    count +=1
    print(f"extract feature file {count} done!!")

class sift_ICM_extract:
  def __init__(self):
    self.sift = cv.xfeatures2d.SIFT_create(100)
  def extract(self, img):
    return cal_sift_CMI_totalimage(img)
  def descriptorSize(self):
      return 24
# path_img = "/media/lmtruong/01D557A8C9369F20/Work/JPEGImages"
# path_output = "/media/lmtruong/01D557A8C9369F20/Work/Feature_extract/sift_ICM_extract"
# sift_ICM_extract(path_img, path_output)