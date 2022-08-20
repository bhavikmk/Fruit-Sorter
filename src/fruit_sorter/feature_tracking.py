import cv2
import numpy as np
import matplotlib.pyplot as plt

file = "/home/bhavik/projects/cv/img/apple4.jpg"
img = cv2.imread(file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray.dtype) # uint8

gray = np.float32(gray) # Convert img to float32

print(gray.dtype) # float32

########## Harris corners

dst = cv2.cornerHarris(gray, 2, 3, 0.04) # img, blocksize, ksize, k

ret, dst = cv2.threshold(dst,0.001*dst.max(),255,0)

dst = np.uint8(dst)

cv2.imshow('dst', dst)

######## Shi thomasi corner detector

corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img, (x,y), 3, 255, -1)

cv2.imshow('dst', img)

######### SIFT (scale invarient Feature transform)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp = sift.detect(gray,None)

# kp, des = sift.detectAndCompute(gray,None)

# print(des)

img=cv2.drawKeypoints(gray,kp,img)

cv2.imshow('dst', img)
cv2.waitKey(3000)

########## SURF (Speeded up robust features)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(400)

kp, des = surf.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, kp, None,(0,255,0),4)

cv2.imshow("keypoints", img)
cv2.waitKey(3000)

########## FAST Algorithm

fast = cv2.FastFeatureDetector_create()

fast.setNonmaxSuppression(0)

kp = fast.detect(img,None)
img = cv2.drawKeypoints(img, kp , None, (0,255,0))

cv2.imshow('fast', img)
cv2.waitKey(3000)

########### BRIEF 

star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

kp = star.detect(img, None)
kp, des = brief.compute(img, kp)

print(des.shape)

########### ORB 

orb = cv2.ORB_create()
kp = orb.detect(img, None)
kp, des = orb.compute(img, kp)
img = cv2.drawKeypoints(img, kp, None, (0,255,0),0)

cv2.imshow('orb', img)
cv2.waitKey(3000)

########### Matching

img1 = img2 = img

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.show()