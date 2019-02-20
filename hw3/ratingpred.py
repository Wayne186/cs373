import numpy as np

# Input: number of labels k
#        vector theta of d rows, 1 column
#        vector b of k-1 rows, 1 column
#        vector x of d rows, 1 column
# Output: label (1 or 2 ... or k)
def run(k,theta,b,x):
    # Your code goes here
    label = 0;
    for i in range(len(b)):
        if ( np.dot(theta, x) < b[i]):
            label = i + 1
            break
        else:
            label = k
    return label