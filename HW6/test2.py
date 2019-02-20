import pcalearn
import pcaproj
import numpy as np
np.set_printoptions(precision=4)
X = np.array([[-2, 2, 0],
  [-3, -1.5, -2],
  [-1, 1, 4],
  [1, -0.5, 5],
  [2, 0, -2]])
mu, Z = pcalearn.run(2,X)
print mu
print Z

X_test = np.array([[-4, 5, 1],
  [2, 1, 4],
  [-4, 5, 7.5],
  [-9.5, 0, 0]])
P_test = pcaproj.run(X_test,mu,Z)
print P_test
