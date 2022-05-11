import cv2
import time
import numpy as np
import PoseEstimatingModel as pm
pTime = 0
cap = cv2.VideoCapture("AI Project Images/sport.mp4")

detector = pm.PoseDetector()
dir = 0
count = 0
while True:
    success, img = cap.read()
    #img = cv2.imread("AI Project Images/test.PNG")
    img = detector.findPose(img, False)
    lmlist = detector.getPosition(img, False)
    if len(lmlist)!=0:
        #Right Angle
        angle =detector.findAngles(img, 12,14,16)
        per = np.interp(angle, (6,77), (0,100))
        bar = np.interp(angle, (6,77), (400,100))
        print(per)
        #Left Angle
        #detector.findAngles(img, 11, 13,15)

        #Check for the dumbell culs
        if per == 100:
            if dir == 0:
                count+=0.5
                dir+=1
        if per == 0:
            if dir == 1:
                count+=0.5
                dir+=0
           
        #Draw Bar
        cv2.rectangle(img, (1100,100), (1175,400), (0,255,0),3)
        cv2.rectangle(img, (1100,int(bar)), (1175,400), (0,255,0),cv2.FILLED)        
        cv2.putText(img, f'{int(per)} %', (1100,75), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0), 4)

        #Draw Curl Count
        cv2.rectangle(img, (0,200), (250,500), (0,255,0),cv2.FILLED)        
        cv2.putText(img, str(int(count)), (90,450), cv2.FONT_HERSHEY_PLAIN, 15, (255,0,0), 15)                

    #print(lmlist)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)