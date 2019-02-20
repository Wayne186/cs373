import numpy as np
import numpy.linalg as la

# Input: number of features F
#        numpy matrix X, with n rows (samples), d columns (features)
# Output: numpy vector mu, with d rows, 1 column
#         numpy matrix Z, with d rows, F columns
def run(F,X):
    # Your code goes here
    n = len(X);
    d = len(X[0])
    mu = np.zeros((d,1))
    for i in range(d):
        sum = 0;
        for t in range(n):
            sum = sum + X[t][i]
        mu[i] = sum/n
    for t in range(n):
        for i in range(d):
            X[t][i] = X[t][i] - mu[i]


    U, s, Vt = la.svd(X, False)

    g = np.zeros(F)
    for i in range(F):
        if (s[i] > 0):
            g[i] = 1 / s[i]
        else:
            g[i] = s[i]

    W = np.zeros((F,d))

    for i in range(F):
        W[i] = Vt[i]

    Z = np.dot(W.T, np.diag(g))

    return (mu, Z)
