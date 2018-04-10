import cv2
import numpy as np
import time

#Functions here grab the image sequence and print all 4 metrics for each feature detector

def SURF(initImg, processImg):
    t0 = time.time()
    surf = cv2.SURF(400)
    kp, des = surf.detectAndCompute(initImg,None)
    t1= time.time()
    speedPF = t1-t0
    numOF = len(kp)
    if (numOF != 0):
        speedPKF = speedPF/numOF
    else:
        speedPKF = speedPF/1

    kp2, des = surf.detectAndCompute(processImg,None)
    numFF = len(kp2)

    print (speedPF, speedPKF, KeyValueComp(numOF, numFF))
def FAST(initImg, processImg):
    t0 = time.time()
    fast = cv2.FastFeatureDetector(40, False)
    kp = fast.detect(initImg, None)
    t1 = time.time()
    speedPF = t1-t0
    numOF = len(kp)
    if (numOF != 0):
        speedPKF = speedPF/numOF
    else:
        speedPKF = speedPF/1

    kp2 = fast.detect(processImg, None)
    numFF = len(kp2)

    print (speedPF, speedPKF, KeyValueComp(numOF, numFF))
def SIFT(initImg, processImg):
    t0 = time.time()
    sift = cv2.SIFT(400)
    kp, des = sift.detectAndCompute(initImg, None)
    t1= time.time()
    speedPF = t1-t0
    numOF = len(kp)
    if (numOF != 0):
        speedPKF = speedPF/numOF
    else:
        speedPKF = speedPF/1

    kp2, des = sift.detectAndCompute(processImg,None)
    numFF = len(kp2)

    print (speedPF, speedPKF, KeyValueComp(numOF, numFF))

def ORB(initImg, processImg):
    t0 = time.time()
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(initImg, None)
    t1= time.time()
    speedPF = t1-t0
    numOF = len(kp)
    if (numOF != 0):
        speedPKF = speedPF/numOF
    else:
        speedPKF = speedPF/1

    kp2, des = orb.detectAndCompute(processImg,None)
    numFF = len(kp2)

    print (speedPF, speedPKF, KeyValueComp(numOF, numFF))

def MSER(initImg, processImg):
    t0 = time.time()
    mser = cv2.FeatureDetector_create('MSER')
    kp = mser.detect(initImg)
    t1= time.time()
    speedPF = t1-t0
    numOF = len(kp)
    if (numOF != 0):
        speedPKF = speedPF/numOF
    else:
        speedPKF = speedPF/1

    kp2 = mser.detect(processImg)
    numFF = len(kp2)

    print (speedPF, speedPKF, KeyValueComp(numOF, numFF))
#This function is called when key values are known, but Percent of Tracked Features
#and Features Count Deviation are not known
def KeyValueComp(numOF, numFF):

    perTrackFeature = numFF - numOF

    if (numOF != 0):
        featureCountDeviation = (numOF - numFF) / numOF
    else:
        featureCountDeviation = (numOF - numFF) / 1

    return (perTrackFeature, featureCountDeviation)

# Uses opencv to load images
initImg = cv2.imread('1.jpg', 0)
cv2.imshow('initial image', initImg)
processImg = cv2.imread('2.jpg', 0)
cv2.imshow('process image', initImg)

SURF(initImg, processImg)
FAST(initImg, processImg)
SIFT(initImg, processImg)
ORB(initImg, processImg)
MSER(initImg, processImg)
