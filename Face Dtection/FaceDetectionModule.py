import cv2
import mediapipe as mp
import time


class FaceDetection():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon


        self.myFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.myFaceDetection.FaceDetection(0.75)

    def findFaces(self, face, draw=True):

        faceRGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(faceRGB)
        bboxs=[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
            #mpDraw.draw_detection(face, detection)
                #print(id, detection)
                #print(detection.score)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = face.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                    int(bboxC.width * iw), int(bboxC.height * ih)

                bboxs.append([id, bbox, detection.score])
                if draw:    
                    img = self.fancyDraw(face, bbox)  
                    cv2.putText(face, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
        return face, bboxs

    def fancyDraw(self, face, bbox, l=30, t=10, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h


        cv2.rectangle(face, bbox, (255,0,255), rt) 
        #Top Left x,y
        cv2.line(face, (x, y), (x+l, y), (255,0,255), t)
        cv2.line(face, (x, y), (x, y+l), (255,0,255), t)

        #Top Left x1,y
        cv2.line(face, (x1, y), (x1-l, y), (255,0,255), t)
        cv2.line(face, (x1, y), (x1, y+l), (255,0,255), t)

        #Bottom Left x,y
        cv2.line(face, (x, y1), (x+l, y1), (255,0,255), t)
        cv2.line(face, (x, y1), (x, y1-l), (255,0,255), t)

        #Bottom Left x1,y1
        cv2.line(face, (x1, y1), (x1-l, y1), (255,0,255), t)
        cv2.line(face, (x1, y1), (x1, y1-l), (255,0,255), t)
        return face


def main():
    cap = cv2.VideoCapture("Videos/face1.mp4")
    pTime=0
    detector = FaceDetection()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Face", img)
        cv2.waitKey(10)











if __name__ == "__main__":
    main()    