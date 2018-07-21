import numpy as np
import random
import math


class LogisticRegression(object):

    def __init__(self):
        self.w = None
        self.ws = None

    def sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def loss(self, X_batch, y_batch, it):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.
        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################

        loss = np.sum((y_batch * np.log(self.sigmoid(X_batch.dot(self.ws[:, it])))) +
                      (1 - y_batch) *
                      np.log(1 - self.sigmoid(X_batch.dot(self.ws[:, it])))) - 0.005 * np.sum(np.square(self.ws[:, it]))
        gradient = np.empty(len(self.ws[:, it]))
        # 偏置项位于样本特征的最后一列
        gradient[-1] = np.sum(y_batch - self.sigmoid(X_batch.dot(self.ws[:, it])))
        for i in range(0, len(self.ws[:, it]) - 1):
            gradient[i] = (y_batch - self.sigmoid(X_batch.dot(self.ws[:, it]))).dot(X_batch[:, i])
        return -loss / len(y_batch), - gradient / len(X_batch)
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def train(self, X, y, learning_rate=1e-3, num_iters=1000,
              batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape

        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        loss_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            batch_index = np.random.choice(num_train, batch_size, False)
            X_batch = X[batch_index]
            y_batch = y[batch_index]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w = self.w - learning_rate * grad
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        temp = self.sigmoid(X.dot(self.ws))
        y_pred = [np.argmax(temp[i]) for i in range(len(temp))]
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
                   batch_size=200, verbose=True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        """
        num_train, dim = X.shape
        if self.ws == None:
            self.ws = np.empty(shape=(dim, 10))
            for i in range(10):
                self.ws[:, i] = 0.0001 * np.random.rand(dim)

        loss_history = {}
        # 训练十个二分类器
        for i in range(10):
            loss_history[i] = []
            for it in range(num_iters):

                batch_index = np.random.choice(num_train, batch_size, False)
                X_batch = X[batch_index]
                y_batch = y[batch_index]
                # 每一个分类器中，标签正确的设置为1，错误的设置为0
                y_batch = np.array(y_batch == i).astype(int)
                loss, grad = self.loss(X_batch, y_batch, i)
                # 当loss下降到一定程度时，将学习率降低
                if loss < 0.2:
                    learning_rate = 1e-7
                loss_history[i].append(loss)
                self.ws[:, i] = self.ws[:, i] - learning_rate * (grad + (0.005 * self.ws[:, i]) / len(y_batch))
                if verbose and it % 2000 == 0:
                    print('class %d iteration %d / %d: loss %f' % (i, it, num_iters, loss))

        return loss_history
