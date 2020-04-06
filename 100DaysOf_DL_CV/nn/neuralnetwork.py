import numpy as np 
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):

        #initialize the list of weight and store the network 
        #architecture and learning rate"""
        self.W = []
        self.layers = layers 
        #list of integers which represents the actual architecture of the feedforward
        #network. For example, a value of [2;2;1] would imply that our first input layer has two nodes,
        #our hidden layer has two nodes, and our final output layer has one node."""
        self.alpha = alpha
        
        #we start looping over the number of layers in the network (i.e., len(layers)),
         #   but we stop before the final two layer (we’ll find out exactly why later in the explantation of this
          #  constructor)."""
        for i in np.arange(0, len(layers)-2):
            # randomly initialize a weight matrix connecting the
            # number of nodes in each respective layer together,
            # adding an extra node for the bias
            w = np.random.randn(layers[i]+1, layers[i+1]+1)
            self.W.append(w / np.sqrt(layers[i]))
            # the last two layers are a special case where the input
            # connections need a bias term but the output does not
        w = np.random.randn(layers[-2]+1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        #construct and return a string that represents the network
        #architecture
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))
    
    def sigmoid(self, x):
        #compute and return the sigmoid activation value for a 
        #given input value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        #compute the derivative of the sigmoid function
        #ASSUMING that "x" has already been passed through the 'sigmoid'
        #function
        return x * (1-x)

    def fit(self,X,y,epochs=1000, displayUpdate=100):
        #insert a column of 1's as the last entry in the feature
        # matrix -- this little trick allows us to treat the bias
        # #as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]
        losses = []
        #loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            #loop over each individual data point and train
            #network on it
            for (x, target) in zip(X,y):
                self.fit_partial(x, target)
            loss = self.calculate_loss(X, y)
            losses.append(loss)
            #check to see if we should display a training update
            if epoch==0 or (epoch + 1) % displayUpdate == 0:
                #loss = self.calculate_loss(X, y)
                #losses.append(loss)
                print("[INFO] epoch = {}, loss = {:.7f}".format(
                    epoch + 1, loss
                ))
        print(len(losses))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, epoch+1), losses)
        plt.title("Training loss")
        plt.xlabel("Epoch #")
        plt.ylabel("loss")
        plt.show()
 
    def fit_partial(self, x, y):
        #construct a list of output activations for each layer
        #as our data points flow through the network;
        A = [np.atleast_2d(x)]

        #FEEDFORWARD
        #loop through the layers in the network
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])

            out = self.sigmoid(net)

            #what was done here is this:
            # h = X*W + b
            # out = sigmoid(h)
            #We will then append the output to the list of activatins
            A.append(out)
            

        #BACKPROPAGATION
        # the first phase of backpropagation is to compute the
        # difference between our *prediction* (the final output
        # activation in the activations list) and the true target
        # value
        error = A[-1] - y

        #applying the chain rule to build our list of deltas, D. The deltas will be
        #used to update our weight matrices, scaled by the learning rate alpha. The first entry in the deltas
        #list is the error of our output layer multiplied by the derivative of the sigmoid for the output value

        D = [error * self.sigmoid_deriv(A[-1])]

        for layer in np.arange(len(A)-2,0,-1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
            # since we looped over our layers in reverse order we need to
                # reverse the deltas
        D = D[::-1]

        #WEIGHT UPDATE PHASE
        #loop over the layers
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])


    def predict(self, X, addBias=True):
        #initialize the output prediction as the input features
        p = np.atleast_2d(X)

        #check to see if the bias column should be added
        if addBias:
            #insert a column of 1's to the last entry of the feature matrix
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        #X = np.c_[X, np.ones((X.shape[0]))]
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        #losses = []
        loss = 0.5 * np.sum(predictions - targets)**2
        #losses.append(loss)
        return loss


    