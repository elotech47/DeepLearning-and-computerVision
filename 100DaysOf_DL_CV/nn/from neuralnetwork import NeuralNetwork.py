from neuralnetwork import NeuralNetwork
import numpy as np
#xonstruct the XOR dataset

X = np.array([[0,0,1], [1,0,1], [1,1,0], [0,1,1]])
y = np.array([[0], [0], [1], [0]])

nn = NeuralNetwork([3,3,1], alpha = 0.5)
nn.fit(X, y, epochs = 20000)


for (x, target) in zip(X,y):
    #make a prediction on the datapoint and display the result
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(
        x, target[0], pred, step
    ))
