import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def InitParameter(x,MaxTime,LearningRate):
    NumberOfFeature = x.shape[1]
    NumberOfSamples = x.shape[0]
    w = np.zeros((NumberOfFeature, 1))
    b = 0
    maxTime = MaxTime
    learningRate = LearningRate
    return NumberOfSamples,NumberOfFeature,w,b,maxTime,learningRate

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def LogisticRegression(x,y,w,b,MaxTime,LearningRate,NOS):
    x=x
    y=y
    w=w
    b=b
    NumberOfSamples=NOS
    maxTime=MaxTime
    learningRate=LearningRate
    for i in range(maxTime):
        LinearOutput = np.dot(w.T, x.T) + b
        SigOutput = Sigmoid(LinearOutput)
        error = SigOutput - y.T
        dw = np.dot(x.T, error.T) / NumberOfSamples
        db = np.sum(error) / NumberOfSamples
        w = w - learningRate * dw
        b = b - learningRate * db
    return w,b

if __name__=='__main__':
    dataset = pd.read_csv('Iris_data.csv', header=None)
    x = dataset.iloc[:, 0:2].values
    y = dataset.iloc[:, -1].values
    NumberOfSamples, NumberOfFeature, w, b, maxTime, learningRate=InitParameter(x,2000,0.1)
    w,b=LogisticRegression(x,y,w,b,maxTime,learningRate,NumberOfSamples)
    print("权重w:")
    print(w)
    print("偏置b:")
    print(b)
    plt.figure(figsize=(8, 6))
    plt.title('Iris_data_LogisticRegression')
    x_1 = dataset.iloc[:50, :2].values
    x_2 = dataset.iloc[51:100, :2].values
    plt.scatter(x_1[:, 0], x_1[:, 1], marker='o', c='g')
    plt.scatter(x_2[:, 0], x_2[:, 1], marker='x', c='r')
    K = -w[0][0] / w[1][0]
    B = -b / w[1][0]
    X = np.arange(0, 5, 0.1)
    Y = K * X + B
    plt.plot(X, Y)

    dataset1 = pd.read_csv('data_1.csv', header=None)
    x1 = dataset1.iloc[:, 0:2].values
    y1 = dataset1.iloc[:, -1].values
    #print(y1[3])
    NumberOfSamples1, NumberOfFeatur1, w1, b1, maxTime1, learningRate1 = InitParameter(x1, 100000, 0.0016)
    w1, b1 = LogisticRegression(x1, y1, w1, b1, maxTime1, learningRate1, NumberOfSamples1)
    print("权重w1:")
    print(w1)
    print("偏置b1:")
    print(b1)
    plt.figure(figsize=(8, 6))
    plt.title('Data_1_LogisticRegression')
    '''
    count_0 = 0
    count_1 = 0
    for i in range(NumberOfSamples1):
        if y1[i]==0:
            count_0=count_0+1
        else:
            count_1=count_1+1
    print(count_1,count_0)
    '''
    x_1_1=np.zeros((NumberOfSamples1,NumberOfFeatur1))
    x_1_0 = np.zeros((NumberOfSamples1, NumberOfFeatur1))
    for i in range(NumberOfSamples1):
        if y1[i]==1:
            x_1_1[i][0] = x1[i][0]
            x_1_1[i][1] = x1[i][1]
        else:
            x_1_0[i][0]=x1[i][0]
            x_1_0[i][1] = x1[i][1]
    '''
    for i in range(NumberOfSamples1):
        if x_1_1[i][0]==0:
            np.delete(x_1_1,i,0)
        elif x_1_0[i][0]==0:
            np.delete(x_1_0,i,0)
    '''
    plt.scatter(x_1_1[:,0],x_1_1[:,1],marker='o',c='b')
    plt.scatter(x_1_0[:, 0], x_1_0[:, 1], marker='x', c='y')
    K1 = -w1[0][0] / w1[1][0]
    B1 = -b1/w1[1][0]
    X1 = np.arange(20, 100, 1)
    #print(B1)
    Y1 = K1 * X1 + B1
    plt.plot(X1, Y1)

    plt.show()