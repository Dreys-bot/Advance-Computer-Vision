import cv2
import mediapipe as mp
import numpy as np
import time
import os
import HandTrackingModel as htm

########################"
brushThickness = 15
eraserThickness = 100
#######################


folderPath = "PaintImages"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[0]
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
xp, yp = 0,0
imgCanvas = np.zeros((720, 1280,3), np.uint8)
detector = htm.handDetector(detectionCon=0.85)
drawColor = (255, 0, 255)
while True:
    #Import image
    success, img = cap.read()
    img = cv2.flip(img,1)


    #Find Hand Landmarks
    img = detector.findHands(img)
    lmlist= detector.findPosition(img, draw=False)
    if len(lmlist) !=0:

        print(lmlist)

        #tip of index and middle fingers
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]


        #Check which finger are up
        fingers = detector.findHandUp()
        print(fingers)

        # If Selection Mode - Two Fingers are up
        if fingers[2] and fingers[3]:
            xp, yp = 0, 0
            print("Selection Mode")
            #Cheking for the click
            if y1 < 119:
                if 250<x1<450:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 550<x1<750:
                    header = overlayList[1]
                    drawColor = (0, 255, 0)
                elif 800<x1<950:
                    header = overlayList[2]
                    drawColor = (255, 0, 0)
                elif 1050<x1<1200:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 30), (x2, y2 + 30), drawColor, cv2.FILLED)
        # If drawing mode - Index Finger is up
        if fingers[2] and fingers[3] == False:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            print("Drawing Mode")
            if xp ==0  and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp,yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv= cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img=cv2.bitwise_or(img,imgCanvas)

    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow('Paint', img)
    cv2.waitKey(1)