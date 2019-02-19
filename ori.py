#From the console, run the following
#pip install numpy
#pip install scipy
#pip install scikit-learn
#pip install matplotlib

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as mp
from pylab import show
from sklearn.model_selection import cross_val_score

data = np.loadtxt("data.csv")

#shuffle the data and select training and test data
np.random.seed(100)
np.random.shuffle(data)


features = []
digits = []


for row in data:
    if(row[0]==1 or row[0]==5):
        features.append(row[1:])
        digits.append(str(row[0]))

#select the proportion of data to use for training
numTrain = int(len(features)*.2)

trainFeatures = features[:numTrain]
testFeatures = features[numTrain:]
trainDigits = digits[:numTrain]
testDigits = digits[numTrain:]

#create the model
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html



X = []
Y = []
simpleTrain = []
colors = []

for index in range(len(trainFeatures)):
    
    n = np.mean(trainFeatures[index])
    q = np.var(trainFeatures[index],ddof=1)

    
    X.append(n)#n was 72
    Y.append(q)#q was 88
    simpleTrain.append([trainFeatures[index],trainFeatures[index]])
    if(trainDigits[index]=="1.0"):
        colors.append("b")
    else:
        colors.append("r")

x_min = np.min(X)
x_max = np.max(X)
y_min = np.min(Y)
y_max = np.max(Y)

for i in range(len(X)):
    X[i] = ((2*(X[i]-x_min))/(x_max-x_min))-1
         
for j in range(len(Y)):
    Y[j] = ((2*(Y[j]-y_min))/(y_max-y_min))-1

#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
#this just shows the points
#mp.scatter(X,Y,s=3,c=colors)
#show()

model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
model_2 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

tmp = []
for i in range(len(X)):
    tmp.append([X[i], Y[i]])
model.fit(tmp,trainDigits)

scores = cross_val_score(model, trainFeatures, trainDigits, cv=10)#256 features
scores_2 = cross_val_score(model_2, tmp, trainDigits, cv=10)#2 features

print("scores:\n")
print(scores)
a = 0
for i in range(0,10):
    a = a + (1-scores[i])
print("\nmean of error:\n")
print(a/10.0)

print("\nscores_2:\n")
print(scores_2)
b = 0
for i in range(0,10):
    a = a + (1-scores_2[i])

print("\nmean of error:\n")
print(b/10.0)


xPred_F = []
yPred_F = []
k_record = []

xF = 1
while xF < 50:
    xPred_F.append(xF)
    model_3 = KNeighborsClassifier(n_neighbors=xF, metric='euclidean')
    scores_3 = cross_val_score(model_3, trainFeatures, trainDigits, cv=10)
    c = 0
    for i in range(0,10):
        c = c + (1-scores_3[i])
    yPred_F.append(c/10.0)
    k_record.append(c/10.0)
    xF = xF+2

#mp.scatter(xPred_F,yPred_F,s=3,alpha=.7)
#show()#1.3

model_4 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
scores_4 = cross_val_score(model_4, tmp, trainDigits, cv=10)
model_4.fit(tmp,trainDigits)


xPred = []
yPred = []
cPred = []
    
for xP in range(-100,100):

    xP = xP/100.0
    for yP in range(-100,100):
        yP = yP/100.0
        xPred.append(xP)
        yPred.append(yP)
        if(model_4.predict([[xP,yP]])=="1.0"):
            cPred.append("r")
        else:
            cPred.append("b")



mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.2)
show()
