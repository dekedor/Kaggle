import numpy as np

def sigmoid(Z):
    return 1./(1 + np.exp(-Z))

def costFunction(X, theta, Y):
    Z = np.array([X[i][:] * theta[i] for i in range(len(theta))])
    return np.sum(-1*Y*np.log(sigmoid(np.sum(Z, axis = 0))) -(1-Y)*np.log(1 - sigmoid(np.sum(Z, axis = 0))))
