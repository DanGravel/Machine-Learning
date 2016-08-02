#Check validation using neural networks

The purpose of this project is to validate the cash amount written on a check and compare it to the user imputed amount.
 
## Neural Network 
The dataset used to train the network was the MNIST hand written digit dataset. The set included 60 000 hand written numbers.  A RandomForestClassifier
was used to classify the training images.  The overall accuracy was approximately 94%.

##Perspective Fix
In order to get a clear birds eye view of the check the perspective needed to be fixed.
The first step in doing this was to find the corners of the check.  This was done with a Harris corner detection algorithm. 
Before running the algorithm the image was slightly blurred to help eliminate noise.
After running the corner detection algorithm the corner points are passed to a function that finds the outermost points.
These outer points are the corners of the check.
From these points a transform matrix can be found and applied to the image to fix the perspective.


Below is a sample picture of a check before fixing the perspective
![alt tag](https://raw.githubusercontent.com/DanGravel/Machine-Learning/master/CheckValidation/Sample%20Images/Checks/check1.jpg)

Using the perspective_fix function the follwing image is obtained.
![alt tag](https://raw.githubusercontent.com/DanGravel/Machine-Learning/master/CheckValidation/Sample%20Images/Checks/perspectiveFix1.jpg)
