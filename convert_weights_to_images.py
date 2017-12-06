"""Converts .weights files into .jpg images."""
import os
import numpy as np
from PIL import Image

def main():
    """Main function."""
    if os.path.exists(os.getcwd()+"/weights/"):
        # Array that holds all the weights from every file.
        all_weights = []
        for file in os.listdir(os.getcwd()+"/weights"):
            # Find files with .weights extension.
            if file.endswith(".weights"):
                # Loads data from file into an array.
                weights = np.loadtxt("./weights/" + file)
                # Saving the weights array for this file into all_weights.
                all_weights.append(weights)
        # Find the min and max weights across all files.
        wmin = min(min(weights) for weights in all_weights)
        wmax = max(max(weights) for weights in all_weights)
        count = 0
        for weights in all_weights:
            # Normalize the data.
            if wmax - wmin > 0:
                weights[...] = (weights[...] - wmin) / (wmax - wmin) * 255
            # Reshape the array into a 2D array.
            weights = weights.reshape((100, 100))
            # Convert the array into an image.
            image = Image.fromarray(weights)
            image = image.convert('RGB')
            # Save the image to disk.
            image.save("./weights/HL" + str(count) + ".jpg")
            count += 1

if __name__ == "__main__":
    main()
