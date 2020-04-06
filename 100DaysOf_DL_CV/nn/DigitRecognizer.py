from neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

#load the MNIST dataset and apply min/max scaling 
print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0],
    data.shape[1]))
(trainX, testX, trainY, testY) = train_test_split(data,
    digits.target, test_size = 0.25)

#convert labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

#train network
print("[INFO] training network...")
#Here we can see that we are training a NeuralNetwork with a 64-32-16-10 architecture.
#The output layer has ten nodes due to the fact that there are ten possible output classes for the digits
#0-9.
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs=2000)

#Evaluating model on testing set
print("[INFO] evaluating network...")
predictions = nn.predict(testX)
#o find the class label with the largest probability for each data point, we use the argmax
#function
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))
