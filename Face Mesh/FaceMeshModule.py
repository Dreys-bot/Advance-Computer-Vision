import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode = False, maxFaces = 2, minDetectionCon = 0.5, minTrackCon = 0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        
        self.myFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.myFaceMesh.FaceMesh(max_num_faces=2)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        
    def findFaceMesh(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        # print(results.multi_hand_landmarks)
        faces = []   
        if self.results.multi_face_landmarks:
            
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.myFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                face = []    
                """ Take a information of face"""
                for id, lm in enumerate(faceLms.landmark):
                    #print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    #cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.15, (0, 255, 0), 1)
                    face.append([id, cx, cy])

                    #print(id, cx, cy)
                faces.append(face)        
        return img, faces



def main():
    pTime = 0
    cap = cv2.VideoCapture("Videos/face2.mp4")
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        im, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()