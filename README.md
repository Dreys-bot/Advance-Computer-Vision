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


