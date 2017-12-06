# Gender Classifier
This is an artificial neural network that takes as input images, and outputs the gender of the person in the picture.

### Getting Started
1) Extract original_images.zip. You should now have a folder called "original_images" in the same folder as the .java and .py files.
2) Run: "python preprocecss_images.py". This will crop all the images and convert them into arrays which are stored in a new folder called "images_as_arrays".
3) Run: "javac GenderClassifier.java". This will compile the Java code.
4) Run: "java GenderClassifier". The ANN will now be trained and tested. A new folder named "weights" will be created, which contains all the weights of the trained ANN. Also a file named "GC.predictions" which contains the predictions for the images in "./original_images/Test". Each line in "GC.predictions" corresponds to an image in that folder.
5) (OPTIONAL) Run: "python convert_weights_to_images.py". This will convert the weights of the first hidden layer into images.

#### Prerequisites
You need python 3 with numpy and Pillow modules.

### More Details
1) preprocess_images.py
    1) Run: "python preprocess_images.py"
        1) Crops images from "./original_images" to 100x100 and ajusts the brightness, then it converts the images to arrays and saves those as .txt files in "./images_as_array".
    2) Run: "python preprocess_images.py -s"
        1) The script now also saves the preprocessed images as .jpg files in ./preprocessed_images.
    
2) GenderClassifier.java
    1) Run: "java GenderClassifier -train"
        1) Trains the ANN.
        2) Saves all the weights to "./weights/weightsFile"
        3) Saves the weights for each node in the first hidden layer to individual files in the weights folder.
    2) Run: "java GenderClassifier -test"
        1) Outputs test predictions to "GC.predictions". Each line corresponds to a file in "/original_images/Test" and contains the certainty for each prediction and the label.
        2) If the "-train" argument is absent, it tries to load the weights from "./weights/weightsFile".
    3) Run: "java GenderClassifier -cv"
        1) Performs 5 fold cross validation using the training set.
        2) Can't be used at the same time as "-train" or "-test".
    4) Run: "java GenderClassifier "
        1) Trains then tests the ANN. Equivalent to "java GenderClassifier -train -test".
        
3) convert_weights_to_images.py (requires the ANN to have been trained)
    1) Takes the weights in "./weights" and converts them to grayscale images. Dark pixels correspont to a negative weight, and bright pixels correspond to a positive weight. 