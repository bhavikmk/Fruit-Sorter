import cv2 as cv
import numpy as np 

def getFrame(frame):
    
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)    
    lower_hsv = np.array([36, 25, 25])
    upper_hsv = np.array([70, 255,255])

    mask_hsv = cv.inRange(hsv, lower_hsv, upper_hsv)
    
    
    masked_image_hsv = np.copy(frame) 
    masked_image_hsv[mask_hsv==0] = [0,0,0]

    return masked_image_hsv

def getFeatures(frame):
    # Classification of image based on features
    return None    

def getPositionFruit():
    # Get position and velocity of moving fruit 
    return None

def ReqdPositionLever():
    # Returns required position of lever for putting fruit in desired lane
    return None

def ActuateLever():
    # Returns required signal to actuate servo  
    return None

img = cv.imread('img\green_apple.jpg')
image = np.copy(img)
masked_img = getFeatures(image)
cv.imshow('Apple',masked_img)

cv.waitKey(0)
cv.destroyAllWindows()