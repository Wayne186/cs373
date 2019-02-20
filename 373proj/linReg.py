
import numpy as np
import psutil
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from multiprocessing import pool
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression


def getDigits_Sub(numSample):
    
    digits = load_digits()
    
#    print(digits.DESCR)

    subset = {"data" : digits.data[0:numSample], "target" : digits.target[0:numSample], "target_names" : digits.target_names, "images" : digits.images[0:numSample], "DESCR" : digits.DESCR}

    return subset



def runLogisticRegression(x_train, x_test, y_train, y_test, givenC, result_queue):
    
    print(givenC,"    Started!!")
    
    logisticRegr = LogisticRegression(C=givenC, random_state=0)
#    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)

#    print("x_train:", len(x_train), "    x_test:", len(x_test))
#    print("y_train:", len(y_train), "    y_test:", len(y_test))


    logisticRegr.predict(x_test[0].reshape(1,-1))
    logisticRegr.predict(x_test[0:10])
    predictions = logisticRegr.predict(x_test)
    score = round(logisticRegr.score(x_test, y_test) * 100, 2)
    print(givenC,"    ", score)

    result_queue.put((predictions, score, givenC))



def run():
    
    numSample = 1000
    numCpu = psutil.cpu_count() - 1
    digits = getDigits_Sub(numSample)
    
    
#    print(len(digits))
#    for key, value in digits.items():
#        print("-----------------------------------")
#        print(key)
#        print(len(value))
#        print(value)
#
##    print((digits["data"][0:21]))
#    print("-----------------------------------")

    print("Shape of image data: " , digits["data"].shape)
    print("Shape of label data: ", digits["target"].shape)

    
    x_train, x_test, y_train, y_test = train_test_split(digits["data"], digits["target"], test_size=0.25, random_state=0)
    
    
    processList = []
    result_queue = multiprocessing.Queue()
    lambdaList = [0.001, 0.1, 10, 1000, 10000]
    
    
    if(numCpu >= len(lambdaList)):
        for C in lambdaList:
            p = multiprocessing.Process(target=runLogisticRegression, args=(x_train, x_test, y_train, y_test, C, result_queue))
            processList.append(p)
            p.start()
    else:
        cnt = 0
        for C in lambdaList:
            cnt += 1
            p = multiprocessing.Process(target=runLogisticRegression, args=(x_train, x_test, y_train, y_test, C, result_queue))
            processList.append(p)
            p.start()
                
            if(cnt == numCpu):
                cnt = 0
                for proc in processList:
                    if(proc.is_alive()):
                        proc.join()
                
    for proc in processList:
        if(proc.is_alive()):
            proc.join()


    result = []
    for proc in processList:
        result.append(result_queue.get())


    for r in result:
        predictions = r[0]
        score = r[1]
        givenC = r[2]

        confMatrix = metrics.confusion_matrix(y_test, predictions)
        plt.figure(figsize=(9,9))
        sns.heatmap(confMatrix, annot=True, fmt=".3f", linewidths=.5, square = True);
        plt.xlabel("Predicted Value");
        plt.ylabel("True Value");
        all_sample_title = "Accuracy: " + str(score) + "%"
        plt.title(all_sample_title, size = 15);
        plt.savefig("ConfusionMatrix_C(" + str(round(givenC,2)) + ").png")
#    print(len(result))
#    pool = multiprocessing.Pool(len(lambdaList))
#    pool.map(graphing_function, result)

if __name__ == '__main__':
    run()
