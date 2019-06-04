import numpy as np
import utils


class DecisionStumpEquality:

    def __init__(self):
        pass


    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)    
        
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count) 

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None
        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)
        # Loop over features looking for the best split
        X = np.round(X)

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] == value])
                y_not = utils.mode(y[X[:,d] != value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] != value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):

        M, D = X.shape
        X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] == self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat


class DecisionStumpErrorRate:

    def __init__(self):
        pass

    def fit(self, X, y):
        """ YOUR CODE HERE """
        N, D = X.shape
        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None
        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)
        # Loop over features looking for the best split
        X = np.round(X)

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] > value])
                y_not = utils.mode(y[X[:,d] <= value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] <= value] = y_not
                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):
        """ YOUR CODE HERE """
        M, D = X.shape
        X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] > self.splitValue:
                yhat[m] = self.splitSat 
            else:
                yhat[m] = self.splitNot

        return yhat



"""
A helper function that computes the entropy of the 
discrete distribution p (stored in a 1D numpy array).
The elements of p should add up to 1.
This function ensures lim p-->0 of p log(p) = 0
which is mathematically true (you can show this with l'Hopital's rule), 
but numerically results in NaN because log(0) returns -Inf.
"""
def entropy(p):
    plogp = 0*p # initialize full of zeros
    plogp[p>0] = p[p>0]*np.log(p[p>0]) # only do the computation when p>0
    return -np.sum(plogp)
    
# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?
class DecisionStumpInfoGain(DecisionStumpErrorRate):

    def __init__(self):
        pass

    def fit(self, X, y):
        """ YOUR CODE HERE """
        #Compute INFO GAIN
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None
        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        # Loop over features looking for the best split
        X = np.round(X)
        
        max_info_gain = 0
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] > value])
                y_not = utils.mode(y[X[:,d] <= value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] <= value] = y_not

                #process n_yes
                try:
                    n_yes_div_n = np.bincount(y[X[:,d] > value])[1] / N
                except IndexError:
                    return
                #process n_no
                n_no_div_n = 1 - n_yes_div_n

                # Compute info gain
                y2 = count / y.size

                try:
                    y_yes = np.bincount(y[X[:,d] > value]) / y.size
                    y_no = 1 - y_yes
                except IndexError:
                    y_no = np.bincount(y[X[:,d] <= value]) / y.size
                    y_yes = 1 - y_no

                info_gain = entropy(y2) - (n_yes_div_n * entropy(y_yes)) - (n_no_div_n * entropy(y_no))
                
                # Compare to max_info_gain so far
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):
        """ YOUR CODE HERE """
        M, D = X.shape
        X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] > self.splitValue:
                yhat[m] = self.splitSat 
            else:
                yhat[m] = self.splitNot

        return yhat



