## WEBCAM 

A webcam is a video camera that is connected to a computer or other device, typically via a USB port, and is used to capture and transmit video over the internet. 

## Example program

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

A histogram is a graph used to represent the frequency distribution of a few data points of one variable


## Use of Histogram

The data should be numerical.

A histogram is used to check the shape of the data distribution. 

Used to check whether the process changes from one period to another.

Used to determine whether the output is different when it involves two or more processes.

Used to analyse whether the given process meets the customer requirements.


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


## Example program for first 10 numbers:

num = list(range(10))

This line creates a list called num containing numbers from 0 to 9 using the range() function and then converting it into a list.

previousNum = 0


This initializes a variable called previousNum to 0. This variable will be used to keep track of the previous number in each iteration of the loop.

for i in num:

This is a loop that iterates through each element in the list num.

sum = previousNum + i

This line calculates the sum of the current number (i) and the previous number (previousNum) and assigns it to a variable named sum.


print('Current Number ' + str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum))


This line prints the current number (i), the previous number (previousNum), and their sum (sum) in a formatted string.


previousNum = i


This line updates the previousNum variable to the current number (i) for the next iteration of the loop.


So, when you run this code, it will output the current number, the previous number, and their sum for each number in the num list, effectively showing the cumulative sum of numbers from 0 to 9.



## Output 

Current Number 0Previous Number 0is 0

Current Number 1Previous Number 0is 1

Current Number 2Previous Number 1is 3

Current Number 3Previous Number 2is 5

Current Number 4Previous Number 3is 7

Current Number 5Previous Number 4is 9

Current Number 6Previous Number 5is 11

Current Number 7Previous Number 6is 13

Current Number 8Previous Number 7is 15

Current Number 9Previous Number 8is 17




## Bounding Boxes

Bounding boxes are rectangular regions that are used to locate objects within an image. They are commonly used in computer vision tasks, such as object detection, object localization, and image segmentation.


## uses of Bounding boxes

1.Object Detection

2.Object Localization

3.Object Tracking

4.Region of Interest (ROI) Selection

5.Semantic Segmentation

## example for Bounding Boxes

1.It imports necessary libraries/modules: os, csv, and specific modules from PIL library (Image and ImageDraw).

import os

import csv

from PIL import Image,ImageDraw

2.Defines paths:

csv_file: Path to the CSV file containing bounding box coordinates (filename, xmin, ymin, xmax, ymax).

image_dir: Directory containing the images.

output_dir: Directory where the output images with bounding boxes will be saved.
    
csv_file = "/home/manichandana-sandhaboina/Downloads/7622202030987_bounding_box.csv"
image_dir = "/home/manichandana-sandhaboina/Downloads/7622202030987"
output_dir = "/home/manichandana-sandhaboina/Downloads/7622202030987_with_boxes"
os.makedirs(output_dir, exist_ok=True)


3.Defines two functions:


draw_boxes(image, boxes): Draws bounding boxes on the given image using ImageDraw.Draw.rectangle() function. The boxes parameter is a list of dictionaries containing bounding box coordinates.
def draw_boxes(image, boxes):


    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        draw.rectangle([left, top, right, bottom], outline="red")
    return image

    
crop_image(image, boxes): Crops the given image based on the bounding box coordinates provided in the boxes parameter. Returns a list of cropped images.
    
def crop_image(image, boxes):
    cropped_images = []
    for box in boxes:
        left = int(box['left'])
        top = int(box['top'])
        right = int(box['right'])
        bottom = int(box['bottom'])
        cropped_img = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_img)
    return cropped_images
    
    
4.Creates the output directory if it doesn't exist already.


with open(csv_file, 'r') as file:

5.Reads the CSV file using csv.DictReader.


    csv_reader = csv.DictReader(file)

6.Iterates through each row in the CSV file:

    
    for row in csv_reader:
        image_name = row['filename']
        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = Image.open(image_path)
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        cropped_images = crop_image(image, boxes)
        for i, cropped_img in enumerate(cropped_images):
            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
        full_image_with_boxes = draw_boxes(image, boxes)
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))

        
7. The draw_boxes() function outlines each bounding box in red on the original image, while the crop_image() function crops the image according to the bounding box coordinates, resulting in individual images containing the detected objects.


## Input


![7622202030987_f306535d741c9148dc458acbbc887243_L_491](https://github.com/Manichandanasandaboina/Task/assets/169050542/1541c5d0-5eda-47aa-ab89-062f4f20f187)



## Output 1



![full_7622202030987_f306535d741c9148dc458acbbc887243_L_491](https://github.com/Manichandanasandaboina/Task/assets/169050542/3664009b-011d-4e6b-8bd9-3be1822a99f1)



## Output2


![0_7622202030987_f306535d741c9148dc458acbbc887243_L_493](https://github.com/Manichandanasandaboina/Task/assets/169050542/a0ed5d67-28a3-4b87-9cfe-a47728fa60e4)
















