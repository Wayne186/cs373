import numpy as np
import numpy.linalg as la

# Input: number of samples n
# number of features d
# Output: matrix X of features, with n rows (samples), d columns (features)
# X[i,j] is the j-th feature of the i-th sample
# vector y of scalar values, with n rows (samples), 1 column
# y[i] is the scalar value of the i-th sample
# Example on how to call the function:
# import createlinregdata
# X, y = createlinregdata.run(10,2)
def run(n,d):
	w = 2*np.random.random((d,1))-1
	w = w/la.norm(w)
	X = np.random.normal(0.0,1.0,(n,d))
	y = np.dot(X,w) + 0.25*np.random.normal(0.0,1.0,(n,1))
	return (X, y)