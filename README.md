# Advance-Computer-Vision

The repository contains 5 mini-projects which are built upon majorly two librairies [MediaPipe](https://mediapipe.dev/) and [OpenCV](https://opencv.org/). While Mediapipe offers a number of functionalities to perform hand tracking, pose estimation etc, openCV helps users to interact with this models.

# Hand Tracking
Detecting and tracking 21 3D hand landmarks at around 25 frames per second.

### Used Technology

* Mediapipe: Mediapipe is a machine learning framework developed by Google. It is open source and lightweight. Mediapipe offers many ML solution APIs such as face detection, face mesh, iris, hands, pose, and there are many. 
In this project, I used Mediapipe's Hands Landmark model.
![hand](https://github.com/Dreys-bot/Advance-Computer-Vision/blob/main/hand.png)

* OpenCV: OpenCV helped me to access the webcam

### Structure

#### Image processing
For the completion of my project, we import two libraries "CV2" and "mediapipe". The "CV2" library allows us to access the camera. 

then, I create an object of the class hand which will allow us to locate the points of the hand visualized by the camera. 

In order to locate the images of the webcam, we use the method "cap.read()" provided by the class "VideoCapture". 

The image captured by our webCam must be converted into RGB because the standard of reading an image in OpenCV is of BGR however with the time which evolved, the standard became RGB from where the conversion.

#### Detection of hand
Next, we detect hands in a frame with the help of a "hand.process()" method Once the hands get detected we move further with locating the key points  and then highlighting the dots in the key point.

![image](https://github.com/Dreys-bot/Advance-Computer-Vision/blob/main/Hand_tracking.gif)

### Final Output
Mediapipe drawing_utils provides a method called "draw_landmarks()" which helps us to connect the points (key points) we have detected. Last but not least, we need to show the final output, the final image to the user "cv2.imshow()" method is handy.

![image](https://github.com/Dreys-bot/Advance-Computer-Vision/blob/main/final_output.gif)


# Face Mesh
Finding and tracking 33 3D full-body landmarks from an image or video.

## Used Technologie
For this project, we need to install OpenCV, Numpy and Mediapipe. We also need to import CV2, mediapipe and time for the project base setup.

## Structure

### Import packages
In the file like faceMeshDetection.py, we import all the packages we need.
The "time()" function help to return the current time in seconds since the Epoch.

### Capture the video
After import all thepackages we need, we create the cv2.VideoCapture(video.path): This fucntion configure the video path. We can change it.

### Set frame Rate
We configure frame rate like below:
```python
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
```

### Mediapipe Face Detection
Mediapipe Face Mesh Detection process an RGB image and returns a list of the detected face location data.
```python
myFaceMesh = mp.solutions.face_mesh
faceMesh = myFaceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
```
### cv2.imread(filename) function:
This function is use to loads an image from a file and returns it.

### cv2.cvtColor(src, code[, dst[, dstCn]])
Converts an image from one color space to another. The function converts an input image from one color space to another. In case of a transformation. to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR).

### process(self, image: np.ndarray) function faceDetection
Processes an RGB image and returns a list of the detected face location data. Takes input image An RGB image represented as a NumPy ndarray.

### Show image cv2.imshow() function.
Face Landmarks are 467 points that are tracked with the user’s face. See the following code below.

```python
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
```  
And we obtain this result:

![face](https://github.com/Dreys-bot/Advance-Computer-Vision/blob/main/faceMesh.png)



# Face Detection
Finding and tracking 33 3D full-face  from an image or video.

## Used Technologie
For this project, we need to install OpenCV, Numpy and Mediapipe. We also need to import CV2, mediapipe and time for the project base setup.

## Structure

### Import packages
In the file like faceMeshDetection.py, we import all the packages we need.
The "time()" function help to return the current time in seconds since the Epoch.

### Capture the video
After import all thepackages we need, we create the cv2.VideoCapture(video.path): This fucntion configure the video path. We can change it.

### Set frame Rate
We configure frame rate like below:
```python
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
```

### Mediapipe Face Detection
Mediapipe Face Mesh Detection process an RGB image and returns a list of the detected face location data.
```python
myFaceMesh = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = myFaceDetection.FaceDetection(0.75)
```
### cv2.imread(filename) function:
This function is use to loads an image from a file and returns it.

### cv2.cvtColor(src, code[, dst[, dstCn]])
Converts an image from one color space to another. The function converts an input image from one color space to another. In case of a transformation. to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR).

### process(self, image: np.ndarray) function faceDetection
Processes an RGB image and returns a list of the detected face location data. Takes input image An RGB image represented as a NumPy ndarray.

### Show image cv2.imshow() function.
This function hrlps us to show the image. See the following code below.

```python
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
```  
And we obtain this result:

![face](https://github.com/Dreys-bot/Advance-Computer-Vision/blob/main/face.png)
    
    
