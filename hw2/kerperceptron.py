import numpy as np
import K as K

# Input: number of iterations L
#numpy matrix X of features, with n rows (samples), d columns (features)
#    X[i,j] is the j-th feature of the i-th sample
#numpy vector y of labels, with n rows (samples), 1 column
#    y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha of n rows, 1 column

def run(L,X,y):
    # Your code goes here
    a = np.zeros(len(X))

    for i in range(L):
        for t in range(len(X)):
            sum = 0;
            for j in range(len(X)):
                k = K.run(X[j], X[t])
                sum = sum + a[j]*y[j]*k
            if (y[t]*sum) <= 0:
                a[t] = a[t] + 1

    return a