import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
myHands = mp.solutions.hands
hands = myHands.Hands()
mpDraws = mp.solutions.drawing_utils

cTime =0
pTime=0

while True:
    success, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            """ Take a information of hand"""
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id==20:
                    cv2.circle(frame, (cx, cy), 25, (255,0,255), cv2.FILLED)
                
            mpDraws.draw_landmarks(frame, handLms, myHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)        

    """Detect if they have hand on the camera"""
    #print(results.multi_hand_landmarks)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)
