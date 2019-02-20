import pcalearn
import pcaproj
import numpy as np
np.set_printoptions(precision=4)
X = np.array([[-2, 2, 0, -2, 2, 0, 4],
  [-3, -1.5, -2, 6, 5, 1, 4],
  [-1, 1, 4, 0, 5, -4, 5],
  [1, -0.5, 5, -9, -9, 0, 0],
  [2, 0, -2, -4.5, 3, 3, 1]])
mu, Z = pcalearn.run(3,X)
print mu
print Z
X_test = np.array([[-4, 5, 1, -4, -4, 0, 1],
  [2, 1, 4, -9.5, -4, 0, 1]])
P_test = pcaproj.run(X_test,mu,Z)
print P_test
