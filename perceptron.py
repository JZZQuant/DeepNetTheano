import numpy as np
import time as ti
import pandas as pd
import pickle

print "loading data from pickle"
sparse_set = pickle.load( open( "data\\save.p", "rb" ) )

predict_x=sparse_set[np.isnan(sparse_set["out"])]

train_unshuffle=sparse_set[~np.isnan(sparse_set["out"])]
train_shuffle=train_unshuffle.iloc[np.random.permutation(len(train_unshuffle))].reset_index()

train=train_shuffle[20:]
test=train_shuffle[:20]

weights = np.random.rand(3,len(test["feature"][6]))
learning=0.1
epochs = 40
max_acc=0
max_weights=0
momentum=0

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+np.exp(-item)))
    y=np.multiply(np.array(a),np.array([1,1,1]))
    normalize=np.sum(y)
    if normalize !=0:
        return y/normalize
    else :
        return np.array([0.33,0.33,0.33])

print " training begins"
for i in range(epochs):
    print "     training phase"
    row_iterator = train.iterrows()
    for j, row in row_iterator:
        output=sigmoid(np.dot(weights,row["feature"]))
        output[int(row["out"])]=output[int(row["out"])]-1
        #predicted=np.argmax(output)
        #if int(row["out"])==predicted:
            #output=np.zeros(3)
        #else :
            #output=np.zeros(3)
            #output[int(row["out"])]=1
            #output[predicted]=1
        delta=-1*output
        a=row["feature"].reshape(1,1575)
        b=delta.reshape(3,1)
        weights= weights+ learning*np.multiply(b,a) + 0.01 * momentum
        momentum = learning*np.multiply(b,a) + 0.01 * momentum
    test_iterator = test.iterrows()
    accuracy=0
    print "     testing phase"
    for k, row in test_iterator:
        prediction=sigmoid(np.dot(weights,row["feature"]))
        output=np.argmax(prediction)
        if int(row["out"]) == int(output):
            accuracy=accuracy+0.05
        if accuracy >= max_acc:
            max_acc=accuracy
            max_weights=weights
    print "     accuracy after epoch "+ str(i) + ": " + str(accuracy)
    print 


weights=max_weights
predict_iterator = predict_x.iterrows()
x=[]
for l, row in predict_iterator:
    prediction=sigmoid(np.dot(weights,row["feature"]))
    x.append(prediction) 
    
final=pd.DataFrame(x)
final.index=predict_x.index+1
final.to_csv("output.csv")