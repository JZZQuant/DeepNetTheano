import os 
import pickle
import pandas as pd
import numpy as np

csvs=[files for files in os.listdir(".") if files.endswith("csv") and not files.startswith("sampl")]
frames=[pd.read_csv(csv) for csv in csvs]
frames[4]["fault_severity"]=np.nan
frames[4]=pd.concat(frames[4:6])
frames=frames[0:5]

for frame in frames:
    frame.ix[:,1]=pd.Categorical(frame.ix[:,1]).codes
    
def arrayvector(pos,size):
    basevector=np.zeros(size)
    basevector[pos-1]=1
    return basevector

for frame in frames:
    l=max(frame.ix[:,1])
    frame.ix[:,1]=frame.ix[:,1].apply(arrayvector,args=(l,))
    
frames[1].ix[:,1]=frames[1].ix[:,2]*frames[1].ix[:,1]
frames[1]=frames[1].ix[:,0:2]  

big=[]
for frame in frames:
    big.append(frame.groupby('id').sum())
big[4]=frames[4].groupby('id').first()
pre_matrix = pd.concat(big, axis=1, join_axes=[big[0].index])

df=[]
for i in range(1,len(pre_matrix)+1):
    df.append(np.concatenate(pre_matrix.ix[i,0:5]))
results=np.array(pre_matrix.ix[:,5])

sparse=pd.concat([pd.Series(df,name="feature"), pd.Series(results,name="out")], axis=1)

import pickle
pickle.dump( sparse, open( "save.p", "wb" ) )