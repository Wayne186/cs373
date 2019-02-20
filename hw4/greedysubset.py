# Input: number of features F
# numpy matrix X of features, with n rows (samples), d columns (features)
#     X[i,j] is the j-th feature of the i-th sample
# numpy vector y of scalar values, with n rows (samples), 1 column
#     y[i] is the scalar value of the i-th sample
# Output: numpy vector of selected features S, with F rows, 1 column
# numpy vector thetaS, with F rows, 1 column
#     thetaS[0] corresponds to the weight of feature S[0]
#     thetaS[1] corresponds to the weight of feature S[1]
#     and so on and so forth

def run(F,X,y):
    # Your code goes here
    S = set()
    for i in range(F):
        for j in range(len(S)):

    return (S, thetaS)