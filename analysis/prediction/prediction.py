import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

nd1=pd.read_csv("alldata5.csv")
nd1=nd1.append(pd.read_csv("alldata10.csv"))
nd1=nd1.append(pd.read_csv("alldata14.csv"))
nd1=nd1.sort_values("Match_id").reset_index(drop=True)

data3=pd.read_csv("data3")
data4=pd.read_csv("data4")
data3=data3.append(data4)
data3=data3.sort_values("Match_id").reset_index(drop=True)

X=nd1.drop(columns=["Result","Match_id"])
Y=nd1["Result"]
Y=Y.replace("Radiant",0)
Y=Y.replace("Dire",1)
X2=data3.drop(columns=["Result","Match_id"])
Y2=data3["Result"]
Y2=Y2.replace("Radiant",0)
Y2=Y2.replace("Dire",1)

X=X[["Player_1_item1","Player_1_item2","Player_1_item3","Player_1_item4","Player_1_item5","Player_1_backpack1","Player_1_backpack2","Player_1_backpack3","Player_2_item1","Player_2_item2","Player_2_item3","Player_2_item4","Player_2_item5","Player_2_backpack1","Player_2_backpack2","Player_2_backpack3","Player_3_item1","Player_3_item2","Player_3_item3","Player_3_item4","Player_3_item5","Player_3_backpack1","Player_3_backpack2","Player_3_backpack3","Player_4_item1","Player_4_item2","Player_4_item3","Player_4_item4","Player_4_item5","Player_4_backpack1","Player_4_backpack2","Player_4_backpack3","Player_5_item1","Player_5_item2","Player_5_item3","Player_5_item4","Player_5_item5","Player_5_backpack1","Player_5_backpack2","Player_5_backpack3","Player_6_item1","Player_6_item2","Player_6_item3","Player_6_item4","Player_6_item5","Player_6_backpack1","Player_6_backpack2","Player_6_backpack3","Player_7_item1","Player_7_item2","Player_7_item3","Player_7_item4","Player_7_item5","Player_7_backpack1","Player_7_backpack2","Player_7_backpack3","Player_8_item1","Player_8_item2","Player_8_item3","Player_8_item4","Player_8_item5","Player_8_backpack1","Player_8_backpack2","Player_8_backpack3","Player_9_item1","Player_9_item2","Player_9_item3","Player_9_item4","Player_9_item5","Player_9_backpack1","Player_9_backpack2","Player_9_backpack3","Player_10_item1","Player_10_item2","Player_10_item3","Player_10_item4","Player_10_item5","Player_10_backpack1","Player_10_backpack2","Player_10_backpack3"]]

l1=[]
l2=[]

for i in X2.itertuples():
  l1=[]

  for j in range(1,137):
    if j in i[1:6]:
      l1.append(1)
    else:
      l1.append(0)
  for j in range(1,137):
      if j in i[6:11]:
        l1.append(1)
      else:
        l1.append(0)  

  
  l2.append(l1)
  

Xoh=pd.DataFrame(l2)

X3=pd.concat([X,Xoh[:26000]],axis=1)

xtr,xtest,ytr,ytest= train_test_split(Xoh,Y2,test_size=0.2, random_state=1)
xtr=xtr.reset_index(drop="True")
ytr=ytr.reset_index(drop="True")
xtest=xtest.reset_index(drop="True")
ytest=ytest.reset_index(drop="True")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8,input_dim=xtr.shape[1],activation='tanh'),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=["accuracy"])

model.summary

history=model.fit(xtr,ytr,batch_size=1000,epochs=100,validation_data=(xtest,ytest))



