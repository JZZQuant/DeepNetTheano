import pandas as pd
from PIL import Image
from numpy import *
import functools
import operator
import numpy as np

class markhov:
    def __init__(self):      
        neareset=pd.read_csv("data\\train.csv")
        neareset["labels"]=neareset["labels"].apply(self.labeltovector)
        grouped=neareset.groupby(["labels"])
        aggregate=list((k, v["business_id"].count()) for k, v in grouped)
        neareset=pd.DataFrame(aggregate, columns=["labels","count"])
        x=pd.DataFrame()
        x["labels"]=range(0,512)
        x["count"]=np.repeat(1,512)
        neareset=pd.merge(x,neareset,on="labels",how="left")
        neareset.fillna(0, inplace=True)
        neareset["count"]=neareset["count_y"]+neareset["count_x"]
        neareset=neareset[["labels","count"]]
        neareset["count"]=neareset["count"]/sum(neareset["count"])
        self.neareset=neareset
        
    def labeltovector(self,label):
        #remember the label orders are reverse
        if type(label)!=str: return label
        results=label.split(' ')
        results = map(int, results)
        num=sum([2**i for i in results])
        #binary='{0:09b}'.format(num)
        #return map(int, list(binary))
        return int(num)

    def likely(self,p):
        likelihood=0
        fit=[]
        for i in range(0,512):
            binary='{0:09b}'.format(i)
            myarray= map(int, list(binary))
            x=np.abs(np.repeat(1,9) - myarray -p)
            t=1
            for y in np.nditer(x):
                t=t*y
            like=self.neareset["count"][i]*t
            if like > likelihood:
                likelihood=like
                fit=myarray
        return [fit]