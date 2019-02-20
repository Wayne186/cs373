import numpy as np
import scipy
import cvxopt
cvxopt.solvers.options['show_progress'] = False


def test_1(a):
    from ratingprank import run as rprun
    import sol_ratingprank as sol_rp
    import sol_ratingpred as sol_pred
    import numpy as np
    L=[a['Q1_L'][0][0][0][0],a['Q1_L'][0][1][0][0],a['Q1_L'][0][2][0][0]]
    dataX=[a['Q1_X'][0][0],a['Q1_X'][0][1],a['Q1_X'][0][2]]
    datay=[a['Q1_y'][0][0],a['Q1_y'][0][1],a['Q1_y'][0][2]]
    totalX=len(dataX)
    totalMatch = 0
    totalTest = 0
    vector_format = 0
    score = 0.0
    for t in range(0,totalX):
        try:
            X,y=dataX[t],datay[t]
            k=int(max(datay[t]))
            (n,dim) = np.shape(X)
            trainSize = int(n*4/5)
            theta1,b1 = rprun(L[t],k,X[0:trainSize],y[0:trainSize])
            theta2, b2 = sol_rp.run(L[t],k,X[0:trainSize],y[0:trainSize])
            if(np.shape(theta1)==(dim,1)):
                vector_format1=1
            elif (np.shape(theta1)==(1,dim)):
                vector_format1=2
                theta1=theta1.reshape(dim,1)
            elif (np.shape(theta1)==(dim,)):
                vector_format1=3
                theta1=theta1.reshape(dim,1)
            else:
                return 0,vector_format
            
            if(np.shape(b1)==(k-1,1)):
                vector_format2=1
            elif (np.shape(theta1)==(1,k-1)):
                vector_format2=2
                theta1=theta1.reshape(k-1,1)
            elif (np.shape(theta1)==(k-1,)):
                vector_format2=3
                theta1=theta1.reshape(k-1,1)
            else:
                return 0,vector_format1,vector_format2
            
            match = 0
            for i in range(trainSize,n):
                label1 = int(sol_pred.run(k,theta1,b1,X[i]))
                label2 = int(sol_pred.run(k,theta2,b2,X[i]))
                match+=1.0- (abs(float(label1-label2)) / float(max(k-label2, label2-1)))
            totalMatch+=match
            totalTest+=(n-trainSize)
            score+=(float(match)/(n-trainSize))
        except:
            return 0,0
        if  vector_format1==1 and vector_format2==1:
            score_format=0
        else:
            score_format=-0.25
    return 4.0*score/totalX,score_format


def test_2(a):
    import sol_ratingsvm as sol_rt
    from ratingpred import run as rpredrun
    import sol_ratingpred as sol_pred
    
    dataX=[a['Q2_X'][0][0],a['Q2_X'][0][1],a['Q2_X'][0][2]]
    datay=[a['Q2_y'][0][0],a['Q2_y'][0][1],a['Q2_y'][0][2]]
    totalX=len(dataX)
    totalMatch = 0
    totalTest = 0
    vector_format=0
    score=0.0
    for t in range(0,totalX):
        X,y=dataX[t],datay[t]
        k=int(max(y))
        (n,dim) = np.shape(X)
        trainSize = int(n*4/5)
        theta1, b1 = sol_rt.run(k,X[0:trainSize],y[0:trainSize])
        theta2  = np.copy(theta1)
        b2= np.copy(b1)
        shapes1 = [(dim,1),(1,dim),(dim,)]
        shapes2 = [(k-1,1),(1,k-1),(k-1,)]
        correct_shapes=[]
        found_shape = False
        vector_formatj=0
        vector_formati=0
        vector_formatl=0
        for j in shapes1:
            vector_formatj+=1
            for i in shapes2:
                vector_formati+=1
                for l in shapes1:
                    try:
                        vector_formatl+=1
                        res = rpredrun(k,theta1.reshape(j),b1.reshape(i),X[trainSize].reshape(l))
                        correct_shapes=[j,i,l]
                        found_shape=True
                    except:
                        found_shape=False
                    if found_shape==True:
                        break
                if (found_shape==True):
                    break
        try:
            match = 0
            for i in range(trainSize,n):
                label1 = rpredrun(k,theta1.reshape(correct_shapes[0]),b1.reshape(correct_shapes[1]),X[i].reshape(correct_shapes[2]))
                label2 = sol_pred.run(k,theta2,b2,X[i])
                match+=1.0- (abs(float(label1-label2)) / float(max(k-label2, label2-1)))
            totalMatch+=match
            totalTest+=(n-trainSize)
            score += (float(match) / (n - trainSize))
        except:
            return 0,0
        if (correct_shapes[0][0]==dim) and correct_shapes[1][0]==k-1  and  correct_shapes[2][1]==dim:
            score_format=0
        else:
            score_format=-0.25
    return 2.0*score/totalX,score_format


def test_3(a):    
    import numpy as np
    from ratingsvm import run as rtrun
    import sol_ratingsvm as sol_rt
    import sol_ratingpred as sol_pred
    
    dataX=[a['Q3_X'][0][0],a['Q3_X'][0][1],a['Q3_X'][0][2]]
    datay=[a['Q3_y'][0][0],a['Q3_y'][0][1],a['Q3_y'][0][2]]
    totalX=len(dataX)
    totalMatch = 0
    totalTest = 0
    vector_format = 0
    score = 0.0
    for t in range(0,totalX):
        try:
            X,y=dataX[t],datay[t]
            k=int(max(datay[t]))
            (n,dim) = np.shape(X)
            trainSize = int(n*4/5)
            theta1,b1 = rtrun(k,X[0:trainSize],y[0:trainSize])
            theta2, b2 = sol_rt.run(k,X[0:trainSize],y[0:trainSize])
            if(np.shape(theta1)==(dim,1)):
                vector_format1=1
            elif (np.shape(theta1)==(1,dim)):
                vector_format1=2
                theta1=theta1.reshape(dim,1)
            elif (np.shape(theta1)==(dim,)):
                vector_format1=3
                theta1=theta1.reshape(dim,1)
            else:
                return 0,vector_format
            
            if(np.shape(b1)==(k-1,1)):
                vector_format2=1
            elif (np.shape(theta1)==(1,k-1)):
                vector_format2=2
                theta1=theta1.reshape(k-1,1)
            elif (np.shape(theta1)==(k-1,)):
                vector_format2=3
                theta1=theta1.reshape(k-1,1)
            else:
                return 0,vector_format1,vector_format2
            
            match = 0
            for i in range(trainSize,n):
                label1 = int(sol_pred.run(k,theta1,b1,X[i]))
                label2 = int(sol_pred.run(k,theta2,b2,X[i]))
                match+=1.0- (abs(float(label1-label2)) / float(max(k-label2, label2-1)))
            totalMatch+=match
            totalTest+=(n-trainSize)
            score+=(float(match)/(n-trainSize))
        except:
            return 0,0
        if  vector_format1==1 and vector_format2==1:
            score_format=0
        else:
            score_format=-0.25
    return 4.0*score/totalX,score_format



def gradeHW_3(stdID=""):
    import scipy.io
    print("Grading "+stdID)
    data=scipy.io.loadmat('hw3data.mat')
    gradeFile = "hw3_grades.csv" 
    prob1,prob1_dim=0,0
    prob2,prob2_dim=0,0
    prob3,prob3_dim=0,0
    with open(gradeFile,"a") as gf:
        try:
            prob1,prob1_dim=test_1(data)
            # pass
        except:
            pass
        try:
            prob2,prob2_dim=test_2(data)
            # pass
        except:
            pass
        try:
            prob3,prob3_dim=test_3(data)
            pass
        except:
            pass
        score=(prob1+prob2+prob3+min(prob1_dim,prob2_dim,prob3_dim))
        score=str.format("%.1f"%score)
        print("Problem 1: %f\nFormat error deducted: %d"%(prob1,prob1_dim))
        print("Problem 2: %f\nFormat error deducted: %d"%(prob2,prob2_dim))
        print("Problem 3: %f\nFormat error deducted: %d"%(prob3,prob3_dim))
        gf.write("%s,%s,%s,%s,%s,%s,%s,%s\n"%(str(stdID), str(prob1),str(prob1_dim),str(prob2),str(prob2_dim),str(prob3),str(prob3_dim),score))


if __name__ == "__main__":
    np.random.seed(26)
    import sys
    gradeFile = "hw3_grades.csv" 
    import os
    exists = os.path.isfile(gradeFile)
    if not os.path.isfile(gradeFile):
        with open(gradeFile,"a") as gf:
            gf.write("id,problem1,problem1_dim,problem2,problem2_dim,problem3,problem3_dim,score\n")
    if(len(sys.argv)==2):
        gradeHW_3(sys.argv[1])
    elif(len(sys.argv)==1):
        gradeHW_3()

