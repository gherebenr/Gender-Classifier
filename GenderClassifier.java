// ANN Gender Classifier
// Robert Ghereben

import java.io.*;
import java.util.*;

public class GenderClassifier {

    private static class Image {
        public ArrayList<Double> pixels = new ArrayList<Double>();
        public String fileName;
        public double gender = 0;  // 1 = male, 0 = female
        // Set gender when constructing object.
        public Image(double g) {
            this.gender = g;
        }
        // Read data from file and save it into an array.
        public Image readImage(File file) {
			this.fileName = file.getName();
            try {
                FileReader fr = new FileReader(file);
                BufferedReader br = new BufferedReader(fr);
                Scanner s = new Scanner(br);
                while(s.hasNextInt()) {
                    pixels.add((double)s.nextInt() / 255);
                }
                s.close();
                br.close();
            } catch (FileNotFoundException e) {
                System.out.println("Can't open file '" + file.getName() + "'");
            } catch (IOException e) {
                e.printStackTrace();
            }
            return this;
        }
        // Pixels array getter.
        public ArrayList<Double> getPixels(){
            return this.pixels;
        }
        // File name getter.
        public String getFileName(){
            return this.fileName;
        }
        // Gender getter.
        public double getGender(){
            return this.gender;
        }
    }

    public static class ANN {
        private int trainingIterations = 10;
        private double alpha = 0.9;
        private double nodeValues[][];
        private double weights[][][];
        private double errors[][];
        // Constructor. Takes in an array of ints. The size of the array is the total number
        // of layers for the ANN, and each element is the number of nodes in that layer.
        public ANN(int numNodesPerLayer[]) {
            // Check if there are at least 3 elements in the array. (input, hidden, output)
            if (numNodesPerLayer.length >= 3) {
                // Initializing the weights, nodeValues and errors arrays to the appropriate sizes.
	            weights = new double[numNodesPerLayer.length - 1][][];
	            nodeValues = new double[numNodesPerLayer.length][];
	            errors = new double[numNodesPerLayer.length - 1][];
	            for (int i = 0; i < numNodesPerLayer.length; i++) {
	                if (i != numNodesPerLayer.length - 1) {
	                    weights[i] = new double[numNodesPerLayer[i] + 1][numNodesPerLayer[i + 1]];
	                }
                    nodeValues[i] = new double[numNodesPerLayer[i]];
                    if(i > 0){
                        errors[i - 1] = new double[numNodesPerLayer[i]];
                    }
	            }
            }
        }

        // Normalize k.
        private double sigmoid(double k) {
            return (double) 1 / (1 + Math.exp(-k));
        }

        // Reset the weights.
        public void resetANNWeights() {
            Random rand = new Random();
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    for (int k = 0; k < weights[i][j].length; k++) {
                        // Setting the weights of pixels with value 0 to 0 so that the 
                        // images formed from the weights don't look like TV static.
                        // Otherwise they are set to random values (-1,1).
                        if(i == 0 && j < nodeValues[i].length && nodeValues[i][j] == 0){
                        	weights[i][j][k] = 0;
                        } else {
                            weights[i][j][k] = rand.nextDouble() * 2 - 1;
                        }
                    }
                }
            }
        }

        // Calculate the errors for each node.
        public void calculateErrors(double trueGenderValue) {
            // Calculate the error for the output layer.
            for (int i = 0; i <  nodeValues[nodeValues.length - 1].length; i++) {
                errors[errors.length - 1][i] = 
                    nodeValues[nodeValues.length - 1][i] - trueGenderValue;
            }
            // Calculate the error for the hidden layers.
            for (int i = nodeValues.length - 2; i > 0; i--) {
                for (int j = 0; j < nodeValues[i].length; j++) {
                    double errorSum = 0;
                    for (int k = 0; k < nodeValues[i+1].length; k++) {
                    	errorSum += weights[i][j][k] * errors[i][k];
                    }
                    errors[i-1][j] = nodeValues[i][j] * (1 - nodeValues[i][j]) * errorSum;
                }
            }
        }

        // Update the weights based on the error.
        public void updateWeights() {
            for (int i = weights.length - 1; i >= 0; i--) {
                for (int j = 0; j < weights[i].length; j++) {
                    for (int k = 0; k < weights[i][j].length; k++) {
                    	if(j == weights[i].length - 1){
                    		weights[i][j][k] -= alpha * errors[i][k];
                    	} else {
                    		weights[i][j][k] -= alpha * errors[i][k] * nodeValues[i][j];
                    	}
                    }
                }
            }
        }

        // Backpropagate error.
        public void errorBackpropagation(double trueGenderValue) {
            calculateErrors(trueGenderValue);
            updateWeights();
        }

        // Returns the output layer after feeding forward the input.
        public double[] feedForward(ArrayList<Double> values) {
            // Setting values in the input layer.
            for (int i = 0; i < nodeValues[0].length; i++) {
                nodeValues[0][i] = values.get(i);
            }
            // Calculating the values of the nodes in the other layers.
            for (int i = 1; i < nodeValues.length; i++) {
                for (int j = 0; j < nodeValues[i].length; j++) {
                    nodeValues[i][j] = weights[i-1][weights[i-1].length-1][j];
                    for(int k = 0; k < nodeValues[i-1].length; k++){
                        nodeValues[i][j] += nodeValues[i-1][k] * weights[i-1][k][j];
                    }
                    nodeValues[i][j] = sigmoid(nodeValues[i][j]);
                }
            }
            return nodeValues[nodeValues.length - 1];
        }

        // Function to train the ANN.
        public ANN trainNetwork(List<Image> trainingSet) {
        	int numMales = 0;
        	int count = 0;
        	double output = 0;
            float trainingAccuracy = 0;
            // Reset the weights.
            resetANNWeights();
            // Count the number of male samples.
        	for(int j = 0; j < trainingSet.size(); j++){
        		if(trainingSet.get(j).getGender() > 0.5){
        			numMales++;
        		}
            }
            // Print what percent of the samples are male.
        	System.out.println(Math.round((float)numMales / (float)trainingSet.size() * 100) + "% males out of " + trainingSet.size() + " training samples.");
            // Start training.
            for (int i = 0; i < trainingIterations; i++) {
            	Collections.shuffle(trainingSet);
                count = 0;
            	for(int j = 0; j < trainingSet.size(); j++){
            		output = feedForward(trainingSet.get(j).getPixels())[0];
                    errorBackpropagation(trainingSet.get(j).getGender());
                    // Count the number of correct guesses.
    				if((output > 0.5 && trainingSet.get(j).getGender() > 0.5) || (output <= 0.5 && trainingSet.get(j).getGender() <= 0.5)){
    					count++;
    				}
                }
                // Print the accuracy every 10 training epochs.
            	if((i+1) % 10 == 0 || i == 0){
                    trainingAccuracy = (float)count / (float)trainingSet.size();
            		System.out.println("Epoch " + (i+1) + ": " + "Training Accuracy = " + trainingAccuracy);
                }
            }
            return this;
        }

        // Perform k fold cross validation.
        public void crossValidation(int k, List<Image> images) {
            double perFoldAccuracy[] = new double[k];
            double accuracySum = 0;
            List<Image> testSet;
            List<Image> trainSet;
            // Calculate the size of the test set.
            int testSetSize = (int) Math.floor((double) images.size() / k);
            // Shuffle the images to prevent the order of the elements in the list
            // from having an impact on the accuracy.
            Collections.shuffle(images);
            // Loop for each fold.
            for (int i = 0; i < k; i++) {
                resetANNWeights();
                // Initialize the test and training sets.
                testSet = images.subList(i * testSetSize, Math.min((i + 1) * testSetSize - 1, images.size() - 1));
                trainSet = new LinkedList<Image>(images);
                // Remove samples that are in the test set from the training set.
                trainSet.removeAll(testSet);
                trainNetwork(trainSet);
                // Count the number of correct labels.
                int count = 0;
                for (Image sample : testSet) {
                    boolean isMale = feedForward(sample.getPixels())[0] > 0.5;
                    if ((isMale && sample.getGender() > 0.5) || 
                        (!isMale && sample.getGender() <= 0.5)) {
                        count++;
                    }
                }
                // Print the accuracy for the fold.
                perFoldAccuracy[i] = (float) count / (float) testSet.size();
                System.out.println("Fold " + i + " Testing Accuracy = " + perFoldAccuracy[i] + "\n");
                accuracySum += perFoldAccuracy[i];
            }
            // Print the average accuracy for all the folds.
            System.out.println("Average Accuracy = " + accuracySum / k);
        }

        // Function to write the predictions for the test set to a file.
        public void outputTestSetPredictions(List<Image> testData) {
            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream("GC.predictions"), "utf-8"))) {
                for (Image img: testData) {
                    double value = feedForward(img.getPixels())[0];
                    // Write the label and how certain the ANN is of its prediction.
                    if (value > 0.5) {
                        bw.write(String.format("%.1f", value*100) + "%\tMALE");
                    } else {
                        bw.write(String.format("%.1f", (1-value)*100) + "%\tFEMALE");
                    }
                    bw.newLine();
                }
            } catch(Exception e) {
                e.printStackTrace();
            }
        }

        // Function to save the weights to a file after training.
        // In case there is a need to -test without -train.
        public void saveWeights() {
            File weightsFolder = new File("./weights");
            if(!weightsFolder.exists()){
                weightsFolder.mkdir();
            }
            // File containing all the weights.
            try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream("./weights/weightsFile")))){
                for (int i = 0; i < weights.length; i++) {
                    for (int j = 0; j < weights[i].length; j++) {
                        for (int k = 0; k < weights[i][j].length; k++) {
                            dos.writeDouble(weights[i][j][k]);
                        }
                    }
                }
                dos.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
            // Write files for only the weights of the input layer. 
            // Used to visualize the weight maps.
            try {
                BufferedWriter bw = null;
                FileWriter fw = null;
                for (int k = 0; k < weights[0][0].length; k++) {
                    fw = new FileWriter("./weights/HL" + k + ".weights");
                    bw = new BufferedWriter(fw);
                    for (int j = 0; j < weights[0].length - 1; j++) {
                        bw.write(Double.toString(weights[0][j][k]));
                        bw.write(" ");
                    }
                    bw.close();
                    fw.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // Load the weights from the weightsFile. Only if there was no -train flag.
        public void loadWeights() {
            try (DataInputStream dos = new DataInputStream(new BufferedInputStream(new FileInputStream("./weights/weightsFile")))){
                for (int i = 0; i < weights.length; i++) {
                    for (int j = 0; j < weights[i].length; j++) {
                        for (int k = 0; k < weights[i][j].length; k++) {
                            weights[i][j][k] = dos.readDouble();
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // Alpha setter.
        public void setAlpha(double a){
            alpha = a;
        }

        // Training iterations setter.
        public void setTrainingIterations(int ti){
            trainingIterations = ti;
        }
    }

    public static void main(String[] args) {
        List<Image> trainSet = new ArrayList<Image>();
        List<Image> testSet = new ArrayList<Image>();
        // Saving the paths the ANN use.
        File fDir = new File("./images_as_arrays/Female/");
        File mDir = new File("./images_as_arrays/Male/");
        File tDir = new File("./images_as_arrays/Test/");
        File weightsFile = new File("./weights/weightsFile");
        // Holds the length of the pixels array. 
        // Used to set the number of nodes in the input layer.
        int numberOfPixels = 0;
        // To make up for the fact that there are more male samples than female ones, 
        // the training set will have four copies of each female sample. This way there will
        // be about 50% female and 50% male samples. This is done to prevent overfitting. 
        int femaleSampleMult = 4;
        // Flags for the command line arguments.
    	boolean crossValidation = false;
        boolean train = false;
        boolean test = false;
        // Setting flags.
        for (int i = 0; i < args.length; i++) {
            if(fDir.exists() && mDir.exists()){
                if (args[i].equals("-train")){
                    train = true;
                }
                if (args[i].equals("-cv")) {
                    crossValidation = true;
                    // Has to be 1 for cross validation, otherwise the accuracy reported
                    // will be wrong.
                    femaleSampleMult = 1;
                }
            }
            if (args[i].equals("-test") && tDir.exists() && (weightsFile.exists() || train)){
            	test = true;
            }
        }
        // If there are no command line arguments, train and test by default.
        if(args.length == 0 && fDir.exists() && mDir.exists() && tDir.exists()){
            test = train = true;
        }
        // Add images to the training set.
        if(train || crossValidation){
            // Add every female sample to the training set.
            File folder = new File(fDir.getAbsolutePath());
            File[] list = folder.listFiles();
            for (File file : list) {
                if (file.isFile() && file.getName().endsWith(".txt")) {
                    Image femaleImg = new Image(0).readImage(file);
                    // Add each sample four times if just training, 
                    // or once if doing cross validation.
                    for (int i = 0; i < femaleSampleMult; i++) {
                        trainSet.add(femaleImg);
                    }
                }
            }
            // Add every male sample to the training set.
            folder = new File(mDir.getAbsolutePath());
            list = folder.listFiles();
            for (File file : list) {
                if (file.isFile() && file.getName().endsWith(".txt")) {
                    trainSet.add(new Image(1).readImage(file));
                }
            }
            numberOfPixels = trainSet.get(0).getPixels().size();
        }
        // Add images to the test set.
        if(test){
            // Add every test sample into the test set.
            File folder = new File(tDir.getAbsolutePath());
            File[] list = folder.listFiles();
            for (File file : list) {
                if (file.isFile() && file.getName().endsWith(".txt")) {
                    testSet.add(new Image(0.5).readImage(file));
                }
            }
            // Sorting the test set so that the lines in the GC.predictions file
            // correspond to the order of the files in the Test folder.
            // Since the file names have an extension, this line removes everything except
            // the numbers and sorts based on those. (e.g. "1.txt" -> 1)
            Collections.sort(testSet, 
                (a,b)->Integer.compare(
                    Integer.parseInt(a.getFileName().replaceAll("[^0-9]", "")), 
                    Integer.parseInt(b.getFileName().replaceAll("[^0-9]", ""))));
            numberOfPixels = testSet.get(0).getPixels().size();
        }
        // Train, test or cross validate based on flags.
        if(numberOfPixels > 0){
            // Create an ANN with 8 nodes in the hidden layer, and one node in the output layer.
            ANN network = new ANN(new int[]{numberOfPixels, 10, 1});
            network.alpha = 0.025;
            network.trainingIterations = 100;
            if (train) {
                System.out.println("\nTraining ANN...");
                network.resetANNWeights();
                network.trainNetwork(trainSet);
                network.saveWeights();
                System.out.println("Training DONE!");
            }
            if (test) {
                System.out.println("\nTesting ANN...");
                if(!train) network.loadWeights();
                network.outputTestSetPredictions(testSet);
                System.out.println("Testing DONE!");
            }
            if(crossValidation && !train && !test){
                System.out.println("\nStarted 5 fold cross validation...\n");
                network.crossValidation(5, trainSet);
                System.out.println("\nCross Validation DONE!");
            }
        }
    }
}