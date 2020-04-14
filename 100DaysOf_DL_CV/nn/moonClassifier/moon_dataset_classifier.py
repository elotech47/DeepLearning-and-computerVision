from neuralnetwork import NeuralNetwork
from plot_decision import plot_decision_boundaries as pltD
#from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt


np.random.seed(24)
X, y = datasets.make_moons(300, noise = 0.50)

#train network
print("[INFO] training network...")
#Here we can see that we are training a NeuralNetwork with a 64-32-16-10 architecture.
#The output layer has ten nodes due to the fact that there are ten possible output classes for the digits
#0-9.
nn = NeuralNetwork([X.shape[1],2,1])
print("[INFO] {}".format(nn))
nn.fit(X,y, epochs=50)

#Evaluating model on testing set
print("[INFO] evaluating network...")
predictions = nn.predict(X)
#o find the class label with the largest probability for each data point, we use the argmax
#function
predictions = predictions.argmax(axis=1)
print(classification_report(y, predictions))

plt.figure(figsize = (10,7))
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)
plt.title('Dataset')
plt.show()

#plt.figure(figsize = (10,7))
pltD(nn, X, y)
plt.title('Decision boundaries')
plt.show()