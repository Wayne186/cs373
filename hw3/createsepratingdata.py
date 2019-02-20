import numpy as np
import scipy.linalg as la
# Input: number of samples n, number of features d, number of labels k
# Output: matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# vector y of labels, with n rows (samples), 1 column
# y[i] is the label (1 or 2 ... or k) of the i-th sample
# Example on how to call the script:
#     import createsepratingdata
#     X, y = createsepratingdata.run(10,2,3)
def run(n, d, k):
    if n < k:
        raise ValueError('n should be at least k')
    X = np.random.random((n, d))
    y = np.zeros((n, 1))
    i = 0
    for r in range(1, k+1):
        j = r*n/k
        X[i:j, 0] = X[i:j, 0] + 1.5*r
        y[i:j] = r
        i = j
    U = la.orth(np.random.random((d, d)))
    X = np.dot(X, U)
    return (X, y)
