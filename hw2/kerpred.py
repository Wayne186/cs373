import K as K

# Input: numpy vector alpha of n rows, 1 column
#numpy matrix X of features, with n rows (samples), d columns (features)
#    X[i,j] is the j-th feature of the i-th sample
#numpy vector y of labels, with n rows (samples), 1 column
#    y[i] is the label (+1 or -1) of the i-th sample
#numpy vector z of d rows, 1 column
# Output: label (+1 or -1)

def run(alpha,X,y,z):
    # Your code goes here
    sum = 0
    for j in range(len(X)):
        k = K.run(X[j], z)
        sum = sum + alpha[j] * y[j] * k
    if sum > 0:
        return 1
    else:
        return -1

