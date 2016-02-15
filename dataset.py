import gzip, cPickle
import numpy as np
import pandas as pd
from PIL import Image
import os

class dataset:
    
    def __init__(self,folder="photos",test_size=64,slash='\\'):
        self.folder=folder
        self.iteration=0
        mapping=pd.read_csv("data\\train_photo_to_biz_ids.csv")
        train=pd.read_csv("data\\train.csv")
        df=pd.merge(mapping,train,on="business_id",how="inner")
        df["photo_id"]=folder+slash+df["photo_id"].astype(str)+".jpg"
        df["labels"]=df["labels"].apply(self.labeltovector)
        df=df[df["photo_id"].apply(os.path.exists)][["photo_id","labels"]]
        self.lump=df
        self.df=df.dropna().iloc[np.random.permutation(len(df))].reset_index()
        self.df["index"]=np.arange(len(self.df))
        if test_size <1:
            self.test_size=int(len(df)*test_size)
        else :
            self.test_size=test_size
        self.train_end=len(df)-self.test_size       
        
    def labeltovector(self,label):
        #remember the label orders are reverse
        if type(label)!=str: return label
        results=label.split(' ')
        results = map(int, results)
        num=sum([2**i for i in results])
        binary='{0:09b}'.format(num)
        return map(float, list(binary))

    def getpix(self,file_name):
        img=Image.open(file_name) 
        pix= np.asarray(img,  dtype='float32')
        del img
        return pix
    
    def next_batch(self,batch):
        end=0
        if self.iteration+batch <= self.train_end: end=self.iteration+batch
        else : end=self.train_end
        x=np.array([y.astype(dtype='float32') for y in self.df[self.iteration:end]["photo_id"].apply(self.getpix)])
        y=np.array([np.array(y).astype(dtype='float32') for y in self.df[self.iteration:end]["labels"]])
        batch=x.reshape((x.shape[0], 3, 32, 32))/255, y
        self.iteration=end
        return batch
    
    def set_back(self):
        self.iteration=0
    
    def test_batch(self,size=64,emit=False):
        index_array = np.arange(self.test_size)
        np.random.shuffle(index_array)
        x=np.array([np.array(self.getpix(self.df["photo_id"][self.train_end + i])).astype(dtype='float32') for i in index_array[:size]])
        y=np.array([np.array(self.df["labels"][self.train_end + i]).astype(dtype='float32') for i in index_array[:size]])
        if emit:
            print "test fire: photo path"+ self.df["photo_id"][self.train_end+index_array[5]] + "actual label: "+ str(self.df["labels"][self.train_end+index_array[5]])
        return x.reshape((x.shape[0], 3, 32, 32))/255,y 
    
    def train_valid_batch(self,size=64):
        index_array = np.arange(self.train_end)
        np.random.shuffle(index_array)
        x=np.array([np.array(self.getpix(self.df["photo_id"][i])).astype(dtype='float32') for i in index_array[:size]])
        y=np.array([np.array(self.df["labels"][i]).astype(dtype='float32') for i in index_array[:size]])
        return x.reshape((x.shape[0], 3, 32, 32))/255,y 
    
    def gotbatch(self,batch):
        return self.iteration+batch <= self.train_end