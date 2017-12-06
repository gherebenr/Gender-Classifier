"""Preprocess .jpg files."""
import sys
import os
from os.path import isfile
import numpy as np
from PIL import Image

def main(argv):
    """Main function."""
    save_images = len(argv) > 0 and argv[0] == "-s"
    if os.path.exists(os.getcwd()+"/original_images/"):
        # Create the preprocessed_images folder if it doesn't exist and is needed.
        if save_images and not os.path.exists(os.getcwd()+"/preprocessed_images/"):
            os.makedirs(os.getcwd()+"/preprocessed_images/")
        # Create the images_as_arrays folder if it doesn't already exist.
        if not os.path.exists(os.getcwd()+"/images_as_arrays/"):
            os.makedirs(os.getcwd()+"/images_as_arrays/")
        # Iterate through every .jpg file in every subfolder in the original_images folder.
        for folder in os.listdir(os.getcwd()+"/original_images/"):
            if not isfile(folder):
                # Create subfolders.
                if not os.path.exists(os.getcwd()+"/images_as_arrays/"+folder):
                    os.makedirs(os.getcwd()+"/images_as_arrays/"+folder)
                if save_images and not os.path.exists(os.getcwd()+"/preprocessed_images/"+folder):
                    os.makedirs(os.getcwd()+"/preprocessed_images/"+folder)
                for file in os.listdir(os.getcwd()+"/original_images/" + folder):
                    if file.endswith(".jpg"):
                        # Load the image.
                        image = Image.open(os.getcwd()+"/original_images/" + \
                            folder + "/" + file).convert('L')
                        # Crop the original image into a 100x100 pixels image.
                        image = image.crop((14, 10, 114, 110))
                        # Normalize the pixels.
                        pixels = np.array(image)
                        maxpixel = pixels.max()
                        pixels[...] = pixels[...] / maxpixel * 255
                        # Save the array as a .txt file.
                        np.savetxt(os.getcwd()+"/images_as_arrays/" + folder + "/" + \
                            file.replace(".jpg", ".txt"), pixels, fmt='%d', delimiter='\t')
                        if save_images:
                            image = Image.fromarray(pixels)
                            # Save the image to disk.
                            image.save(os.getcwd()+"/preprocessed_images/" + folder + "/" + file)

if __name__ == "__main__":
    main(sys.argv[1:])
