import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils 
cap = cv2.VideoCapture("Videos/vid2.mp4")


pTime =0

while True:
    success, video = cap.read()
    vidRGB  = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
    results = pose.process(vidRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(video, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = video.shape
            
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(video, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(video, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('Dance', video)
    cv2.waitKey(20)

