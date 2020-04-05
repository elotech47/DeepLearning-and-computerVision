#import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#make_blobs, a function used to create “blobs" of normally distributed data points
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np 
import argparse

def sigmoid(x):
    """Computes the sigmoid activation value for a 
    given input """
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    """Takes the dot product between our features and weight matrix"""
    preds = sigmoid(X.dot(W))
    #apply a step function to threshold the 
    #outputs to binary
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1
    #return the predictions
    return preds

#construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-e","--epoch", type = float, default = 100,
        help = "number of epochs")
ap.add_argument("-a","--alpha", type = float, default = 0.01,
        help = "learning rate")
args = vars(ap.parse_args())    

#Generate a 2-class classification problem with 1000 data points,
#where each data point is a 2D feature vector
(X,y) = make_blobs(n_samples=1000, n_features=2, centers=2,
    cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

#inserts a columns of ones as the last entry in the feature
#matrix -- this is the bias trick
X = np.c_[X, np.ones((X.shape[0]))]

#partition the data into training and testing splits using 50% of 
#data for training and the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size = 0.5, random_state = 42)

#initialize the weight matrix and the list of losses
print("[INFO training...")
W = np.random.randn(X.shape[1],1)
losses = []

#loop over the desired number of epochs
for epoch in np.arange(0, args["epoch"]):
    preds = sigmoid(trainX.dot(W))

    error = preds - trainY
    loss = np.sum(error**2)
    losses.append(loss)

    gradient = trainX.T.dot(error)

    W += -args["alpha"] * gradient

    #check to see if an update should be displayed
    if epoch==0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch = {}, loss = {:.7f}".format(int(epoch + 1),loss))

#evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

#plot the testing classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
#plt.scatter(testX[:,0], testX[:,1], marker = "o", s = 30)
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)
#plt.plot(testY)

#construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epoch"]), losses)
plt.title("Training loss")
plt.xlabel("Epoch #")
plt.ylabel("loss")
plt.show()
