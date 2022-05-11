import cv2
import mediapipe as mp
import time
import PoseEstimatingModel as pm

pTime = 0
cap = cv2.VideoCapture("Videos/vid2.mp4")
detector = pm.PoseDetector()

while True:
    success, video = cap.read()
    video = detector.findPose(video)
    lmlist = detector.getPosition(video)
    print(lmlist)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(video, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", video)
    cv2.waitKey(20)