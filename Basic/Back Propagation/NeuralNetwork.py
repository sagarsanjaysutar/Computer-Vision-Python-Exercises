import numpy as np

class NeuralNetwork:
    ''' Intializing weight matrix. '''
    def __init__(self, layers, alpha=0.01):
        self.W = [] # Initialize the list of weight matrix.
        self.layers = layers
        self.alpha = alpha

        # Intializing weight matrix by looping all layers but last 2 as we need to define it seperately. 
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1) # + 1 is extra node for bias. 
            self.W.append(w / np.sqrt(layers[i]))

        # The second last layer requires bias but the last does not.
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
    
    ''' Debugging '''
    def __repr__(self):
        return "Neural Network: {}".format("-".join(str(l) for l in self.layers))

    ''' Activation function '''
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    ''' Derived sigmoid: To be used during back propagation '''
    def sigmoid_derived(self, x):
        return x * (1 - x)

    ''' Training function '''
    def fit(self, training_data, train_label, epochs=100, display_rate=100):

        training_data = np.c_[training_data, np.ones((training_data.shape[0]))] # Add a column for bias.

        for epoch in np.arange(0, epochs):
            for (data_point, data_label) in zip(training_data, train_label):
                self.fit_partial(data_point, data_label)
            
            # Displaying o/p in terminal
            if epoch == 0 or (epoch + 1) % display_rate == 0:
                loss = self.calculate_loss(training_data, train_label)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    ''' Calculate loss '''
    def calculate_loss(self, data, labels):
        labels = np.atleast_2d(labels)
        predictions = self.predict(data, addBias = False)
        loss = 0.5 * np.sum((predictions - labels) ** 2) # Compute sum of squared error.
        return loss

    ''' Back Propagation '''
    def fit_partial(self, data_point, data_label): 
        
        # List of o/p activations for each layer as our data point flows through the network.
        # 1st activation is just our input
        A = [np.atleast_2d(data_point)] 

        # Feedforward : Loop over each layer, dot product b/w input data and weight and then pass it to activation fn.
        for layer in np.arange(0, len(self.W)):
            out = self.sigmoid(A[layer].dot(self.W[layer])) # Dot product of input data and weight matrix and then go to next layer
            A.append(out)
        
        # Back propagation : Reverse loop, dot product b/w delta and weight and then multiply by derivative of activation for that layer
        error = A[-1] - data_label # Calculate the difference between predicted value(our last layer) and target value.
        Gradiants = [error * self.sigmoid_derived(A[-1])] # List of delta. First entry is error of o/p layer

        for layer in np.arange((len(A) - 2), 0, -1): # Ignore the last 2 layer as they are already taken into account above.
            gradient = Gradiants[-1].dot(self.W[layer].T)
            gradient = gradient * self.sigmoid_derived(A[layer]) # Derivative of activation for that layer
            Gradiants.append(gradient)
        
        Gradiants = Gradiants[::-1] # Reverse Gradiants, as we looped reversed.

        # Weight update phase
        for layer in np.arange(0, len(self.W)):
            self.W[layer] = self.W[layer] - self.alpha * A[layer].T.dot(Gradiants[layer])

    ''' Prediction '''
    def predict(self, data, addBias = True):

        # 1st prediction is just our input
        p = np.atleast_2d(data) 

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        
        # Forward propagate through network to get final prediction.
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p
            

        
            
        
