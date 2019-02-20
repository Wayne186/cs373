import cvxopt as co
import numpy as np
import K as K

# Input: numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
#numpy vector y of labels, with n rows (samples), 1 column
#y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha of n rows, 1 column

def run(X,y):
    # Your code goes here
    b = np.zeros(len(y))
    f = -1 * np.ones(len(y))
    print f
    A = -1 * np.identity(len(y))
    H = np.zeros( (len(y), len(y)) )

    print H

    for i in range(len(H)):
        for j in range(len(H[0])):
            k = K.run(X[i], X[j])
            H[i][j] = y[i]*y[j]*k

    print H

    alpha = np.array(co.solvers.qp(co.matrix(H, tc='d'), co.matrix(f, tc='d'),co.matrix(A, tc='d'), co.matrix(b, tc='d'))['x'])

    return alpha
