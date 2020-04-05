#import necessary packages
import cv2
import numpy as np
import os

class SimpleDatasetLoader:
    def __init__(self,preprocessors = None):
        #store the image preprocessor
        self.preprocessors = preprocessors


        #If the preprocessors are None, initialize them as an 
        #empty list
        if self.preprocessors is None:
            self.preprocessors = []
        

    def load(self,imagePaths, verbose=-1):
        #initialize the list of the features and labels
        data = []
        labels = []

        #loop over the input images
        for (i,imagePath) in enumerate(imagePaths):
            #Load the image and extract class label
            #assuming that our path as the following format:
            #/path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            #check to see if our preprocessors are not None
            if self.preprocessors is not None:
                #Loop over the preprocessors and apply each
                #to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            #treat our preprocessed image as feature vectors
            #by updating the data list folloed by the labels
            data.append(image)
            labels.append(label)

            #show an update every 'verbose' images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed{}/{}".format(i + 1, len(imagePaths)))

        #return a tuple of the data and the labels
        return (np.array(data),np.array(labels))




