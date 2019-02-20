import createsepdata
import linperceptron
import linprimalsvm
import linpred
import numpy as np


print "Hello Python"
(X, y) = createsepdata.run(10, 2)
print (X, y)


X = np.array([[-0.80415671,  0.50588813],
       [-0.57706297, -0.07384805],
       [-0.49077184,  0.15227998],
       [-0.35103191,  0.03149906],
       [-0.80248863, -0.43501787],
       [-0.36387386,  0.59511482],
       [-0.13357491,  0.61985742],
       [ 0.45245728,  0.96209518],
       [ 0.0073892 ,  1.08794835],
       [-0.35021918,  0.59301836]
])

y = np.array([[ 1.],
       [ 1.],
       [ 1.],
       [ 1.],
       [ 1.],
       [-1.],
       [-1.],
       [-1.],
       [-1.],
       [-1.]
])


print "start theta"

theta = linperceptron.run(10, X, y)
print "result:"
print theta

label = linpred.run(theta, X[0])
print label

theta = linprimalsvm.run(X, y)
print theta

print "pred result:"
for i in range(len(X)):
    label = linpred.run(theta,X[i])
    print ": ", label
