import numpy as np

# Input: number of iterations L
# numpy matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# numpy vector y of labels, with n rows (samples), 1 column
# y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector theta of d rows, 1 column

def run(L, X, y):
    # Your code goes here
    theta = np.zeros(len(X[0]))

    for t in range(L):
        for i in range(len(X)):
            if (np.dot(X[i], theta) * y[i]) <= 0:
                theta = theta + y[i] * X[i]

    theta = theta.reshape((-1, 1))
    return theta
