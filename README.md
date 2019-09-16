# BGsubtractor

Computer vision is an extremely hot field in computer science and we can see multiple uses of computer vision technology around us. In this project we will talk about use of background subtraction method to develop an object detector for automated surveillance system and later talk about the cons of such traditional technique for the given and task and handle it with some state of the art real time object detecting algorithm called YOLO which uses Convolutional Neural Network (CNN) for object classification with extremely high accuracy.
Introduction
Computer Vision is booming field at this point and in the past two decades we have seen some amazing improvement in the world of object detection and classification. The Application of object detection and classification has absolute no limit. In this project our main task is to build automatic surveillance system where cars are detected using traditional outdated technique using “background subtraction “method and then next state of the art technique called “you only look once ” also known as YOLO for object detection and classification.
Procedure
Choice of Language : Python_version3.7 Packages used : cv2 , NumPy , time
Library used for the ‘background subtraction method’ part of the project is OpenCV[1] which is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications to accelerate the use of machine perception in the commercial products.

Background subtraction method :
Background subtraction (BS) is a common and widely used technique for generating a foreground mask ( namely, a binary image containing the pixels belonging to moving objects in the scene) by using static cameras.
As the name suggests, Background subtraction calculates the foreground mask performing a subtraction between the current frame and a background model, containing the static part of the scene or, more in general, everything that can be considered as background as background given the characteristic of observed scene.


Background subtraction principles:
- Background subtraction should segment objects of interest when they first appear ( of reappear) in scene.
- An appropriate pixel-level stationary criterion should be defined. Pixels that satisfy this criterion are declared background and ignored.
- The background model must adapt to both sudden and gradual changes in the background.
Background models should take into account changes at differing spatial scales.
Background subtraction consists of two main steps :
1. Background initialization
2. Background update
In the first step [Background initialization] , an initial model of background is computed, while in the second step that model is updated in order to adapt to possible changes in the scene.
But before you we apply background subtraction method to our image we first have to prepare our individual frames of input video stream
First we get the frame’s dimension to define our Region of interest (ROI).
To create a region of interest (ROI) using basic image manipulation we have to know the entry and exit point of cars. We define input and output lines for cars using Blue and Red Color respectively.
Blue Line will use to start the ROI and red line for the end of ROI.
Define the role of pts variable here
Now that we have selected region of interest (ROI) for our frame we can start applying background subtraction method to extract foreground.
We are
going
to
use
createBackgroundSubtractorMOG2()
function to extract the foreground from the image. The function used here is based on Gaussian Mixture-based foreground segmentation Algorithm[2[.
Detection shadow = True to include shadows of the moving object in the foreground mask.
Our next task is to set up kernels for applying Gaussian blur to the individual frames.
Blurring is surprisingly useful operation in computer vision. It might seem like blurring an image would obscure exactly the information that you’re seeking to extract and foreground in our case.
To the contrary, blurring is similar to taking an average : it combines the value of each pixels with its neighbors. This is really useful for eliminating small variations or noise that might hurt the results of other operations like edge detection, contour finding etc.[3]
We prepare our kernels for blur using NumPy library .
for the purpose of tracking every car detected is treated as an instance of a class.
Class “Car “ contains all the necessary attribute for the creating an instance for the detected car.
Every car is given an ID using getID(self) function for the purpose of tracking the car through multiple of frame. We also use tracks[]. List attribute to append on the tracking coordinates to keep the trajectory of object in terms of pixels position on the frame. Code is implemented to detect cars going both ways i.e. cars going UP the road and car going DOWN the road so for that purpose we have an attribute self.dir which is initialized to None and is set to UP or DOWN after tracking it’s entrance direction in ROI.
Then we create an infinite loop until our grabbed frame return value is maintained to be True.
Then we grab our individual frame from the video using read() function and which return individual frame and a Boolean value equals to True if frames are detected.
Then we apply background subtractor to get our 2 foreground mask
Now we use threshold function to achieve binarization of the of out two foreground mask.
Binarization :
- The idea of thresholding is to further- simplify visual data for analysis.
- At the most basic level thresholding convert everything to white or black based on a threshold value. Example let’s say we want the threshold to be 125 ( our of 255) then everything that was 125 and under will be converted to 0 or black and everything above 125 will be converted to 255, or white.
It is sometime very difficult to choose a threshold value that is compatible to the entire image, and in many cases even impossible.
In our case we choose 200 as our threshold value and therefore everything above threshold value will be converted to white i.e. 255 in pixel value.
Our next step is to apply morphological transformation to our processed frame.
Morphological transformation[4]
- Morphological transformation are some simple operations based on the image shape. It is normally performed on binary images it needs two inputs, one is. Our original image, second one is called structuring element or kernel which decides the nature of operations.
- Two basic morphological operators are Erosion and Dilation
- Erosion : similar to soil erosion. It erodes away the boundaries of foreground object ( always try to keep foreground in white).
Process : kernel slides through the image , and a pixel in the original image ( either 1 or 0 ) will be considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded ( made to zero ) [1 implies pixel is switched on and 0 implies pixel is switched off]
- Dilation : it is opposite of Erosion. Here a pixel element is 1 if at least one pixel under the kernel is 1. So it increases the white region in the image or size of foreground object increases.
For noise removal we apply transformation in the given order . [Erosion followed by dilation also known as opening] to both mask.
Even after noise removal we would have some holes in the foreground object or small black points on the abject so above transformation is followed by another transformation i.e. [Dilation followed by Erosion also known as closing] to close up those gaps in the foreground.
Finding contours :
Contours can be explained simply as a curve joining all the continuous points ( along the boundary ) having same color or intensity. The contours are useful tool for shape analysis and object detection and recognition.
findContours() function is used to find the contour points of given contour.
Here cv2.CHAIN_APPROX_NONE is used to print out all the data points of the contours in our image and cv2.RETR_EXTERNAL will give us external contours of the detected object.
Finding moments :
Contours moments help us to calculate some features like center of mass of the object , area of the object etc. In our case we want to find the centroid of the given contour.
After finding the centroid of the contour we use cv2.boundingRect() function of OpenCV library to draw approximate rectangle around the binary image.
This function is used mainly to highlight the region of interest after obtaining contours from an image.
Tracking :
we take the coordinate of the centroid and keep updating the coordinate of the each Car instance with similar ID with centroid coordinates using updateCoords(x,y) function of the class Car.
After the car exits Region of Interest (ROI) we delete the car instance from the cars[] list.
Now that we have detected the car we put circle around the centroid of the car and a bounding box around the detected moving object.
At the end we use release() function to release the Video frames and De-allocate any associated memory usage using destroyAllWindows() function in OpenCV
Code Optimization: [files not added in repo]
We can improve our FPS simply by creating a new thread that does nothing but poll the camera for new frames while our main thread handles processing the current frame.
In order to accomplish this FPS increase/latency decrease, our goal is to move the reading of frames from a webcam or USB device to an entirely different thread, totally separate from our main Python Script. This allows frames to be read continuously from I/O thread, all while our root thread processes the current frame. Once the root thread has finished processing its frame, it simply needs to grab the current frame from the I/O thread. This accomplished without having to wait for blocking I/O operations.
The first step in implementing our threaded video stream functionality is to define FPS class that we can use to measure our frames per second. This class will help us obtain quantitative evidence that threading does indeed increase FPS.
We’ll then define a WebcamVideoStream class that will access our webcam in a threaded fashion. Finally, we’ll define our driver script, fps_demo.py, that will compare single threaded FPS to multi- thread FPS.


Advantages and Disadvantages
Advantages :
- A different “threshold “ is selected for each pixel
- These pixel-wise “threshold” are adapting by time
- Objects are allowed to become part of the background without destroying the existing background model.
- Provides fast recovery
Disadvantages :
- Cannot deal with sudden, drastic lighting changes
- Initializing the gaussians is important ( median filtering )
- There are relatively many parameters, and they should be selected intelligently
There are many challenges that has to be conquered while suing background subtraction algorithm.
- It must be robust against changes in illumination
- It should avoid detecting non-stationary background objects such as swinging leaves, rain , snow and shadow cast by moving objects.
- Finally , its internal background model should react quickly to changes in


