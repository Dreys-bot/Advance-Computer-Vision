import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/face1.mp4")

myFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = myFaceDetection.FaceDetection(0.75)
pTime =0
while True:
    success, face = cap.read()

    faceRGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(faceRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
           #mpDraw.draw_detection(face, detection)
            #print(id, detection)
            #print(detection.score)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = face.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(face, bbox, (255,0,255), 2)    
             
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(face, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
    cv2.imshow('Face', face)
    cv2.waitKey(10)
