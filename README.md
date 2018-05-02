# Gender Classifier
Neural Network which takes as input images, and outputs the gender of the person in the picture.

### Getting Started
1) Extract "original_images.zip" in the same folder as the .java and .py files.
2) Open a terminal and navigate to the folder where the .java and .py files are.
3) Run the following commands:
	```
	1) python preprocess_images.py         | This will crop all the images and convert them into arrays.
	2) javac GenderClassifier.java          | This will compile the Java code.
	3) java GenderClassifier                | This will train and test the ANN.
	4) python convert_weights_to_images.py | This will convert the weights of the first hidden layer into images.
	```
    
### Prerequisites
1) Python 3 with numpy and Pillow modules. 
2) Java Development Kit.

### More Details
1) ***preprocess_images.py***
    1) ```python preprocess_images.py```
        1) Crops images from "./original_images" to 100x100 and adjusts the brightness.
        2) Converts the images to arrays and saves them as .txt files in "./images_as_array".
    2) ```python preprocess_images.py -s```
        1) The script now also saves the preprocessed images as .jpg files in "./preprocessed_images".
    
2) ***GenderClassifier.java***
    1) ```java GenderClassifier -train```
        1) Trains the ANN.
        2) Saves all the weights to "./weights/weightsFile".
        3) Saves the weights for each node in the first hidden layer to individual files in "./weights/".
    2) ```java GenderClassifier -test```
        1) Outputs test predictions to "GC.predictions". Each line corresponds to a file in "./original_images/Test" and contains the certainty for each prediction and the label.
        2) If the "-train" argument is absent, it tries to load the weights from "./weights/weightsFile".
    3) ```java GenderClassifier -cv```
        1) Performs 5 fold cross validation using the training set.
        2) Can't be used at the same time as "-train" or "-test".
    4) ```java GenderClassifier```
        1) Trains then tests the ANN. Equivalent to "java GenderClassifier -train -test".
        
3) ***convert_weights_to_images.py*** (requires the ANN to have been trained)
    1) ```python convert_weights_to_images.py```
    	1) Takes the weights in "./weights" and converts them to grayscale images where darker pixels correspond to a negative weight, and bright pixels correspond to a positive weight. 
