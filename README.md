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









## Histogram

## Histogram Plotting using OpenCV and Matplotlib

This code reads an image, calculates histograms for each color channel (blue, green, and red), and plots them using OpenCV and Matplotlib.

Installation:

1. Install NumPy, OpenCV, and Matplotlib using pip:

   pip install numpy opencv-python matplotlib

Usage :

1. Run the script plot_histogram.py.
2. Provide the path to the input image as an argument.
3. The script will generate a histogram plot for each color channel and display it.
   
   
Code Explanation:

import numpy as np

import cv2 as cv

from matplotlib import pyplot as plt

- cv.imread(): Reads the input image.
  
- cv.calcHist(): Calculates histograms for each color channel.
  
- plt.plot(): Plots the histograms using Matplotlib.

Read the image:

img = cv.imread('/home/manichandana-sandhaboina/Desktop/experIments/flower.jpg')

This line reads an image file named 'flower.jpg' from the specified path.

Write the image:

cv.imwrite('/home/manichandana-sandhaboina/Desktop/histo.jpg', img)

This line writes the image img to a new file named 'histo.jpg' in the specified path.

Check if the image is read successfully:

assert img is not None, "file could not be read, check with os.path.exists()"

This line checks if the image is successfully read. If the image is not read (i.e., img is None), it raises an assertion error with the message "file could not be read, check with os.path.exists()".

Calculate and plot histogram:

color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

This part calculates the histogram of the image for each color channel (blue, green, and red) separately using cv.calcHist() function. Then it plots the histograms using matplotlib.pyplot.plot() function. Finally, it sets the x-axis limit from 0 to 256 and displays the plot using matplotlib.pyplot.show() function.

## Input 

![flower](https://github.com/Manichandanasandaboina/Task/assets/169050542/c2f9de9c-2ca1-4cee-ba97-6ba41a99947e)


## Output

![Figure_1](https://github.com/Manichandanasandaboina/Task/assets/169050542/b5c7ba78-3e0a-4c02-a341-3d45894ad923)




## Iteration






