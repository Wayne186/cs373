import pcalearn
import pcaproj
import numpy as np
np.set_printoptions(precision=4)
X = np.array([[-3, 2],
  [-2, 1.5],
  [-1, 1],
  [0, 0.5],
  [1, 0]])
mu, Z = pcalearn.run(1,X)
print mu
print Z

X_test = np.array([[-3, 2],
  [-2, 1.5],
  [-1, 1],
  [0, 0.5],
  [1, 0]])
P_test = pcaproj.run(X_test,mu,Z)
print P_test

