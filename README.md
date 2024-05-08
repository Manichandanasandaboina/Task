## WEBCAM 

1.Importing the OpenCV Library:

import cv2 

This line imports the OpenCV library. OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library.

2.Defining a Video Capture Object:

vid = cv2.VideoCapture(0) 

cv2.VideoCapture(0) initializes the camera capture. The argument 0 represents the index of the camera device (usually the webcam). If you have multiple cameras, you can specify a different index to select a different camera.

3.Loop for Capturing and Displaying Frames:

while(True):

This starts an infinite loop, which will continue until a break statement is encountered.

4.Capturing Video Frame by Frame:

ret, frame = vid.read()

This line captures a frame from the video capture object (vid) and stores it in the variable frame.
    
vid.read() reads the next frame from the video stream. The variable ret indicates whether the frame was successfully read.

5.Displaying the Frame:

cv2.imshow('frame', frame)

This line displays the captured frame in a window named 'frame'.

cv2.imshow() is used to display images in OpenCV. The first argument is the window name, and the second argument is the image to be displayed.

6.Checking for Quit Signal:

if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.waitKey(1) waits for a key press for 1 millisecond. If a key is pressed, it returns the ASCII value of the key.

7.Releasing Video Capture Object and Closing Windows:

vid.release()
cv2.destroyAllWindows()

After the loop, vid.release() releases the video capture object.
    
     
cv2.destroyAllWindows() closes all OpenCV windows.


## Output

[Screencast from 08-05-24 11:46:17 AM IST.webm](https://github.com/Manichandanasandaboina/Task/assets/169050542/e620fc60-c5f1-4258-95d2-8385c5a9fd25)



