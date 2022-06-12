import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)

def getFeatures(frame):
    # For binary classification 0-1 (for multiple classification,use 0-n)
    # good_fruit = 0
    # bad_fruit = 1
    frame_category = 0
    
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    lower_hsv = np.array([100,0,0])
    upper_hsv = np.array([160,150,150])

    mask_hsv = cv.inRange(hsv, lower_hsv, upper_hsv)
    masked_image = frame 
    masked_image[mask_hsv==0] = [0,0,0]

    return masked_image

def getPositionFruit():
    return None

def getPositionLever():
    return None

def ReqdPositionLever():
    return None

def ActuateLever():

    # Returns required angle of servo actuator 
    return None

while True:
    rate, frame = camera.read()
    
    masked_image = getFeatures(frame)
    
    cv.imshow('camera',masked_image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()




# -> getFeatures
# -> getPositionFruit
# -> getPositionLever
# -> ReqdPositionLever
# -> ActuateLever