# OCR Math Equation Project

This project is a simple model for recognizing characters in mathematical equations. The goal was to develop a system capable of identifying characters from images of math equations.

## Description
- The project uses the OpenCV library, specifically the `cv2.connectedComponentsWithStats` function, to detect and separate characters.
- The final accuracy of the model is approximately 67%, with potential for improvement through further enhancements.

## How It Works
1. An image of a mathematical equation is provided as input.
2. The `cv2.connectedComponentsWithStats` function is used to isolate individual characters from the image.
3. The developed model recognizes the separated characters.

## Sample Images
![Sample Image 1](img/1.png)  
![Sample Image 2](img/2.png)  

## Requirements
- Python 3.x
- OpenCV (`cv2`)