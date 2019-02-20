import numpy as np
# Input: number of iterations L
# number of labels k
# matrix X of features, with n rows (samples), d columns (features)
#     X[i,j] is the j-th feature of the i-th sample
# vector y of labels, with n rows (samples), 1 column
#     y[i] is the label (1 or 2 ... or k) of the i-th sample
# Output: vector theta of d rows, 1 column
#         vector b of k-1 rows, 1 column

def run(L, k, X, y):
# Your code goes here
    theta = np.zeros(len(X[0]))
    b = np.zeros((k-1, 1))
    for i in range(k-1):
        b[i] = i

    s = 0;
    for i in range(L):
        for t in range(len(X)):
            E = np.zeros((k - 1, 1))
            for l in range(k-1):
                if y[t] > l+1:
                    s = 1
                else:
                    s = -1
                if s * (np.dot(theta, X[t])-b[l]) <= 0:
                    E[l] = s
            print np.transpose(E)
            s = 0
            for l in range(k-1):
                s = s + E[l];
            print s
            if s != 0:
                theta = theta + s * X[t]
            print X[t], theta
            for l in range(k-1):
                b[l] = b[l] - E[l]


    return (theta, b)
