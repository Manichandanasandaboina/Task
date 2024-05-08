## Histogram

# Histogram Plotting using OpenCV and Matplotlib

This code reads an image, calculates histograms for each color channel (blue, green, and red), and plots them using OpenCV and Matplotlib.

## Installation

1. Install NumPy, OpenCV, and Matplotlib using pip:

   pip install numpy opencv-python matplotlib

   
## Usage

1. Run the script `plot_histogram.py`.
2. Provide the path to the input image as an argument.
3. The script will generate a histogram plot for each color channel and display it.

## Code Explanation

- `cv.imread()`: Reads the input image.
- `cv.calcHist()`: Calculates histograms for each color channel.
- `plt.plot()`: Plots the histograms using Matplotlib.

## Example

```python
import cv2 as cv
from matplotlib import pyplot as plt

# Read the image
img = cv.imread('flower.jpg')

# Calculate histograms and plot them
# (Code snippet from the provided code)
plt.show()

## Input




 







