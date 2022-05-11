import cv2
import mediapipe as mp
import time
import math

class PoseDetector():
    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon=0.5, trackCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils 
        
    def findPose(self,video, draw = True):
        vidRGB = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(vidRGB)
        # print(results.multi_hand_landmarks)

        if self.results.pose_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(video, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return video

    def getPosition(self, video, handNo = 0, draw = True):

        self.lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = video.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(video, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return self.lmlist


    def findAngles(self, img, p1,p2,p3, draw = True): 
        #Get the landmark
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]


        #Calculate the angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y2-y1, x2-x1))

        if angle <0:
            angle+=360
        #Draw
        if draw:
            cv2.line(img, (x1,y1), (x2, y2), (255,255,0),3)
            cv2.line(img, (x3,y3), (x2, y2), (255,255,0),3)
            cv2.circle(img, (x2, y2), 15, (0, 255, 255), 2) 
            cv2.circle(img, (x2, y2), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 255, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 255, 255), 2)   
            cv2.putText(img, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return angle    
def main():
    pTime = 0
    cap = cv2.VideoCapture("Videos/vid2.mp4")
    detector = PoseDetector()

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


if __name__ == "__main__":
    main()