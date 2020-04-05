#import necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.SimplePreprocessor import SimplePreprocessor
from pyimagesearch.datasets.SimpleDatasetLoader import SimpleDatasetLoader
from imutils import paths
import argparse

datasets = "C:/Users/Elotech/dev/PyImageSearchCourse/datasets"

#grab the list of images that we will be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(datasets))

#initialize the image preprocessor, load the dataset from disk,
#and reshape the data matrix
sp = SimplePreprocessor(150,150)
sdl = SimpleDatasetLoader(preprocessors = [sp])
(data,labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0],150*150*3))

#show some information on memory consumption of the image
print("[INFO] features matrix: {:.1f}MB".format(
    data.nbytes / (1024 * 1000.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)
#labels = le.transform(labels)
#print(labels)
#partition the data into training and testing splits using 75% of 
# the data for training and the remaining 25% for testing

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state = 42)

#train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier....")
model = KNeighborsClassifier(n_neighbors=5,
    n_jobs =-1)

model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
    target_names=le.classes_))

