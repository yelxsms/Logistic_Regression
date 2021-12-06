import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 装载数据
dataset=pd.read_csv('Iris_data.csv', header=None)
x=dataset.iloc[:,0:2].values
y=dataset.iloc[:,-1].values

# 初始化参数
NumberOfFeature=x.shape[1]
NumberOfSamples=x.shape[0]
w=np.zeros((NumberOfFeature,1))
b=0
maxTime=500
learningRate=0.01

# Sigmoid 逻辑函数
def Sigmoid(x):
    return 1/(1+np.exp(-x))

# 主体
for i in range(maxTime):
    LinearOutput=np.dot(w.T,x.T)+b
    SigOutput=Sigmoid(LinearOutput)
    # 偏差损失
    error=SigOutput-y.T
    # 求梯度，瓜瓜书p55
    dw=np.dot(x.T,error.T)/NumberOfSamples
    db=np.sum(error)/NumberOfSamples
    # 参数更新
    w=w-learningRate*dw
    b=b-learningRate*db

# 输出参数
print("权重w:")
print(w)
print("偏置b:")
print(b)

# 可视化
# 原始数据绘制散点图
x1=dataset.iloc[:50,:2].values
x2=dataset.iloc[51:100,:2].values
plt.scatter(x1[:,0],x1[:,1],marker='o',c='g')
plt.scatter(x2[:,0],x2[:,1],marker='x',c='r')
# 决策边界
K=-w[0][0]/w[1][0]
B=-b/w[1][0]
X=np.arange(0,5,0.1)
Y=K*X+B
plt.plot(X,Y)
plt.show()