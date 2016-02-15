import theano
import theano.tensor as T
import numpy as np
import time as ti
import dataset as d
import markhov as m
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from sklearn import metrics
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()
import pandas as pd
from PIL import Image

ds=d.dataset(test_size=0.15)
ma=m.markhov()
x_gold, labels_gold = ds.test_batch(size=128,emit=False)
# define symbolic Theano variables
x = T.tensor4()
t = T.matrix()

# define model: neural network
def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

def meanfscore(y_pred,y_true):
    return metrics.f1_score(np.array(np.rint(y_true),dtype="int"), np.array(np.rint(y_pred),dtype="int") , average='samples')  

def momentum(cost, params, learning_rate, momentum):
    grads = theano.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = momentum * mparam_i - learning_rate * g
        updates.append((mparam_i, v))
        updates.append((p, p + v))

    return updates

def dropout(x, p=0.):
    if p > 0:
        retain_prob = 1 - p
        x *= srng.binomial(x.shape, p=retain_prob, dtype=theano.config.floatX)
        x /= retain_prob
    return x

def model(x, w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o, p=0.0):
    c1 = T.maximum(0, conv2d(x, w_c1) + b_c1.dimshuffle('x', 0, 'x', 'x'))
    p1 = max_pool_2d(c1, (3, 3),ignore_border=False)

    c2 = T.maximum(0, conv2d(p1, w_c2) + b_c2.dimshuffle('x', 0, 'x', 'x'))
    p2 = max_pool_2d(c2, (2, 2),ignore_border=False)

    p2_flat = p2.flatten(2)
    p2_flat = dropout(p2_flat, p=p)
    h3 = T.maximum(0, T.dot(p2_flat, w_h3) + b_h3)
    p_y_given_x = T.nnet.sigmoid(T.dot(h3, w_o) + b_o)
    return p_y_given_x

w_c1 = init_weights((4, 3, 3, 3))
b_c1 = init_weights((4,))
w_c2 = init_weights((8, 4, 3, 3))
b_c2 = init_weights((8,))
w_h3 = init_weights((8 * 4 * 4, 100))
b_h3 = init_weights((100,))
w_o = init_weights((100, 9))
b_o = init_weights((9,))

params = [w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o]


p_y_given_x_train = model(x, *params, p=0.5)
p_y_given_x_test = model(x, *params, p=0.0)
#[ 0.10636091,  0.07805075,  0.16173595,  0.15493369,  0.08857718,0.10269261,  0.13499186,  0.12473306,  0.047924  ]
y_train = T.switch(p_y_given_x_train > 0.4, 1, 0) 
y_test = T.switch(p_y_given_x_test > 0.4, 1, 0) 

cost = T.mean(T.nnet.binary_crossentropy(p_y_given_x_train, t))
updates = momentum(cost, params, learning_rate=0.012, momentum=0.9)

# compile theano functions
train = theano.function([x, t], cost, updates=updates,allow_input_downcast=True)
predict = theano.function([x], y_test,allow_input_downcast=True)
raw_predict = theano.function([x], p_y_given_x_test,allow_input_downcast=True)

# train model
batch_size = 40
final=0
start=ti.time()
prev_valid_accuracy=0
past_train_accuracy=0
train_accuracy=0
max_valid_accuracy=0
minimax=0
storedparams=0
for i in range(90):
    ds.set_back()
    print "iteration %d" % (i + 1) 
    j=0
    for j in range(0,2600):
    #while ds.gotbatch(batch_size):
        j=j+1
        temp=params
        x_batch,t_batch = ds.next_batch(batch_size)
        cost = train(x_batch , t_batch )
        if j%11==0:
            alpha_train = predict(x_batch)
            train_accuracy = np.mean(meanfscore(alpha_train,t_batch))
            x_valid,t_valid=ds.test_batch()
            alpha_valid=predict(x_valid)
            valid_accuracy = np.mean(meanfscore(alpha_valid,t_valid))
            if valid_accuracy>max_valid_accuracy:
                max_valid_accuracy=valid_accuracy
            # make sure overfitting doesnt happen
            if past_train_accuracy > train_accuracy and valid_accuracy > prev_valid_accuracy:
                params=temp
            else :
                prev_valid_accuracy=valid_accuracy
            past_train_accuracy=train_accuracy
            if valid_accuracy < 0.90*max_valid_accuracy:
                params=temp
    x_test, labels_test = ds.test_batch(size=128,emit=False)
    predictions_test = predict(x_test)
    accuracy = np.mean(meanfscore(predictions_test,labels_test))
    
    #test against a standard gold standarm to see variation in prediction across samples
    predictions_gold = predict(x_gold)
    accuracy_gold = np.mean(meanfscore(predictions_gold,labels_gold))
    
    #print "for a quick check try this image if it all looks cool for a prediction, actual labels:"+str(labels_test[5]) + " observerd labels: " +str(predictions_test[5])
    final=accuracy
    if min(accuracy,accuracy_gold)>minimax:
        minimax=min(accuracy,accuracy_gold)
        storedparams=params
    print "accuracy: %.5f" % accuracy
    print "accuracy on gold standard: %.5f" % accuracy_gold
    print "took time stamp " + str(ti.time()-start)
    start=ti.time()
    print
print "final minimax accuracy " + str(minimax)
params=storedparams
print
print "starting the prediciton phase"

dftest=pd.read_csv("data\\test_photo_to_biz.csv")
#dry run
dftest=dftest
dftest["photo_id"]="test_photos\\"+dftest["photo_id"].astype(str)+".jpg"

def prediction(file_name):
    img=Image.open(file_name) 
    pix= np.asarray(img,  dtype='float32')
    del img
    sample=np.array(pix.astype(dtype='float32')).reshape((1, 3, 32, 32))/255
    outcome=raw_predict(sample)
    return outcome

dftest["labels"]= dftest["photo_id"].apply(prediction)
dftest=dftest[["business_id","labels"]]
grouped=dftest.groupby(["business_id"])
aggregate = list((k, v["labels"].mean()) for k, v in grouped)
dftest=pd.DataFrame(aggregate, columns=["business_id", "labels"])
dftest2=pd.DataFrame(aggregate, columns=["business_id", "labels"])
categorize = lambda value: np.where(value>0.4,1,0)
vectortolabel=lambda label: str(np.where(np.fliplr(label)==1)[1])[1:-1]
dftest["labels"]=dftest["labels"].apply(categorize).apply(vectortolabel)
dftest2["labels"]=dftest2["labels"].apply(ma.likely).apply(vectortolabel)  
#still need to consolidate results using hmm
dftest.to_csv("raw_submission.csv",index =False)
dftest2.to_csv("hmm_submission.csv",index =False)