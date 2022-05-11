import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/face1.mp4")
myFaceMesh = mp.solutions.face_mesh
faceMesh = myFaceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
cTime =0
pTime=0

while True:
    success, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms, myFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

            """ Take a information of face"""
            for id, lm in enumerate(faceLms.landmark):
                #print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                """if id==20:
                    cv2.circle(frame, (cx, cy), 25, (255,0,255), cv2.FILLED)"""

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f'FPS : {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0 ), 3)        

    """Detect if they have hand on the camera"""
    #print(results.multi_hand_landmarks)

    cv2.imshow("FaceMesh", frame)
    cv2.waitKey(10)
