import numpy as np
from numpy import random

class TwoLayersNN (object):
    """" TwoLayersNN classifier """

    def __init__ (self, inputDim, hiddenDim, outputDim):
        self.params = dict()
        self.params['w1'] = None
        self.params['b1'] = None
        self.params['w2'] = None
        self.params['b2'] = None
        #########################################################################
        # TODO: 20 points                                                       #
        # - Generate a random NN weight matrix to use to compute loss.          #
        # - By using dictionary (self.params) to store value                    #
        #   with standard normal distribution and Standard deviation = 0.0001.  #
        #########################################################################
        pass

        self.params['w1'] = random.normal(0, 0.0001, (inputDim, hiddenDim))
        self.params['b1'] = np.zeros(shape=(1, hiddenDim))
        self.params['w2'] = random.normal(0, 0.0001, (hiddenDim, outputDim))
        self.params['b2'] = np.zeros(shape=(1, outputDim))


        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss (self, x, y, reg):
        """
        TwoLayersNN loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to each parameter (w1, b1, w2, b2)
        """
        loss = 0.0
        grads = dict()
        grads['w1'] = None
        grads['b1'] = None
        grads['w2'] = None
        grads['b2'] = None
        #############################################################################
        # TODO: 40 points                                                           #
        # - Compute the NN loss and store to loss variable.                         #
        # - Compute gradient for each parameter and store to grads variable.        #
        # - Use Leaky RELU Activation at hidden and output neurons                  #
        # - Use Softmax loss
        # Note:                                                                     #
        # - Use L2 regularization                                                   #
        # Hint:                                                                     #
        # - Do forward pass and calculate loss value                                #
        # - Do backward pass and calculate derivatives for each weight and bias     #
        #############################################################################
        pass

        N = x.shape[0]
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        c1 = x.dot(w1) + b1
        d1 = np.maximum(c1, 0.01*c1)
        c2 = d1.dot(w2) + b2
        d2 = np.maximum(c2, 0.01*c2)
        L = self.params['w2'].shape[1]
        S = d2.copy()
        S -= np.max(S, axis=1).reshape(N, 1)
        Sp = S[np.arange(N), y].reshape(N, 1)
        Ss = np.sum(np.exp(S), axis=1).reshape(N, 1)
        P = (np.exp(Sp))/Ss
        Li = -1 * (np.log(P))
        loss = np.sum(Li)/N
        loss += reg * (np.sum(w1**2) + np.sum(w2**2))

        Exp_S_p = np.exp(S)
        P = Exp_S_p / np.sum(Exp_S_p, axis=1, keepdims=True)
        Do = P
        Do[range(N), y] -= 1
        dd2 = Do
        Do/=N
        dz2 = dd2.copy()
        dz2[[d2 < 0]] = .01
        dz2[[d2 >= 0]] = 1
        dz2 *= dd2
        grads['w2'] = np.dot(d1.T, dz2)
        grads['w2'] += (reg * (w2) **2)
        dd1 = dz2.dot(w2.T)
        dz1 = dd1.copy()
        dz1[d1<0] = .01
        dz1[d1>=0] = 1
        dz1 *= dd1
        grads['w1'] = np.dot(x.T, dz1)
        grads['w1'] += (reg * (w1**2))
        db2 = np.sum(dz2, axis=0)
        db1 = np.sum(dz1, axis=0)
        grads['b1'] = db1.copy()
        grads['b2'] = db2.copy()


        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, grads

    def train (self, x, y, lr=5e-3, reg=5e-3, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iterations):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (batchSize, D)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################
            pass

            batchID = np.random.choice(np.arange(x.shape[0]), batchSize, replace=False)
            xBatch = x[batchID]
            yBatch = y[batchID]

            loss, grads = self.calLoss(x=xBatch, y=yBatch, reg=reg)

            self.params['w1'] -= lr * grads['w1']
            self.params['b1'] -= lr * grads['b1']
            self.params['w2'] -= lr * grads['w2']
            self.params['b2'] -= lr * grads['b2']

            lossHistory.append(loss)

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # Decay learning rate
            lr *= decay
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print ('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict (self, x,):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Store the predict output in yPred                                    #
        ###########################################################################
        pass

        z1 = x.dot(self.params['w1']) + self.params['b1']
        d1 = np.maximum(z1, .01 * z1)
        z2 = d1.dot(self.params['w2']) + self.params['b2']
        d2 = np.maximum(z2, .01 * z2)
        yPred = np.argmax(d2, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred


    def calAccuracy (self, x, y):
        acc = 0
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################
        pass
        yPred = self.predict(x)
        acc = 100 * (np.mean(y == yPred))


        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return acc



