from neuralnetwork import NeuralNetwork
import numpy as np
from plot_decision import plot_decision_boundaries as pltD
#xonstruct the XOR dataset
import matplotlib.pyplot as plt

X = np.array([[0,1], [0,0], [1,0], [1,1]])
y = np.array([[1], [0], [1], [0]])

nn = NeuralNetwork([2,2,1], alpha = 0.5)
nn.fit(X, y, epochs = 200)


for (x, target) in zip(X,y):
    #make a prediction on the datapoint and display the result
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(
        x, target[0], pred, step
    ))

# plt.figure(figsize = (10,7))
# plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)
# plt.title('Dataset')
# plt.show()

#plt.figure(figsize = (10,7))
pltD(nn, X, y)
plt.title('Decision boundaries')
plt.show()