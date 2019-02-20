import createsepratingdata
import ratingpred
import ratingprank
import numpy as np

(X, y) = createsepratingdata.run(10,2,5)
print (X, y)

(theta, b) = ratingprank.run(10, 5, X, y)
print 'theta = '
print theta
print 'b = '
print b

print "pred result:"
for i in range(len(X)):
    label = ratingpred.run(5, theta, b, X[i])
    print ": ", label