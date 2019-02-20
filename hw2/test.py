import createsepdata
import kerperceptron as kp
import kerpred
import kerdualsvm as ksvm


(X, y) = createsepdata.run(10, 2)
print (X, y)

print "start a"
a = kp.run(10, X, y)
print "result:"
print a

print "pred result:"
for i in range(len(X)):
    label = kerpred.run(a,X,y,X[i])
    print ": ", label

a = ksvm.run(X,y)
print a

print "pred result:"
for i in range(len(X)):
    label = kerpred.run(a,X,y,X[i])
    print ": ", label