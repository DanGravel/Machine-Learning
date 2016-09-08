#Check validation using machine learning

The purpose of this project is to validate the cash amount written on a check and compare it to the user imputed amount.
 
## Classifier
The dataset used to train the classifier was the MNIST hand written digit dataset. The set included 60 000 hand written numbers.  The HOG features of each image was calculated and a LinearSVC was used to fit the data.


##Perspective Fix
In order to get a clear birds eye view of the check the perspective needed to be fixed.
The first step in doing this was to find the corners of the check.  This was done with a Harris corner detection algorithm. 
Before running the algorithm the image was slightly blurred to help eliminate noise.
After running the corner detection algorithm the corner points are passed to a function that finds the outermost points.
These outer points are the corners of the check.
From these points a transform matrix can be found and applied to the image to fix the perspective.


Below is a sample picture of a check before fixing the perspective.
![alt tag](https://github.com/DanGravel/Machine-Learning/blob/master/CheckValidation/Sample%20Images/Checks/check1.jpg)

Using the perspective_fix function the follwing image is obtained.
![alt tag](https://raw.githubusercontent.com/DanGravel/Machine-Learning/master/CheckValidation/Sample%20Images/Checks/perspectiveFix1.jpg)

##Detecting Subregions
To validate the cash amount on a check you have to find the written amount on the check.  This was done using thresholding, morphological operations, and contours. The image was first converted to grayscale.  Next a blackhat operator was used to reveal dark regions on the check. The dark regions on a check is the writing. A Scharr gradient was then used to more accurately select text regions. The threshold of the image was then computed and morphological transformations was used to join gaps in text regions.

The result of the threshold and morphological transformation can be seen below.
![alt tag](https://github.com/DanGravel/Machine-Learning/blob/master/CheckValidation/Sample%20Images/Checks/threshold.png)

Next the contours in the image above were found. The size of each contour was checked to insure that it was not to small or large. This left you with the cash amount on a check and can be seen below.
<p align="center">
 <img src="https://github.com/DanGravel/Machine-Learning/blob/master/CheckValidation/Sample%20Images/Checks/dollars.png")>
</p>
The threshold of the cash amount was calculated.  This can be seen below.
<p align="center">
 <img src="https://github.com/DanGravel/Machine-Learning/blob/master/CheckValidation/Sample%20Images/Checks/dollars_th.png")>
</p>
Next the contours of the threshold image were found.  Contours that were very small were removed since they are not numbers. The contours found can be seen below.
<p align="center">
 <img src="https://github.com/DanGravel/Machine-Learning/blob/master/CheckValidation/Sample%20Images/Checks/number4.png")>
 <img src="https://github.com/DanGravel/Machine-Learning/blob/master/CheckValidation/Sample%20Images/Checks/number3.png">
 <img src="https://github.com/DanGravel/Machine-Learning/blob/master/CheckValidation/Sample%20Images/Checks/number2.png">
 <img src="https://github.com/DanGravel/Machine-Learning/blob/master/CheckValidation/Sample%20Images/Checks/number1.png">
 <img src="https://github.com/DanGravel/Machine-Learning/blob/master/CheckValidation/Sample%20Images/Checks/number0.png">
</p>

Each image was resized to be 28x28, they were then passed to the classifier in neuralNet.py
The result from the classifier was [7][7][7][7]. The last digit was incorrect, this could be due to artifacts in the image caused by neighboring numbers.  Another issue was detecting the decimal point.



