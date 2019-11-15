#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019-4-7 20:36
#@Author: Seasons
#@File  : CNN2.py

#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019-4-7 18:01
#@Author: Seasons
#@File  : CNN.py

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#合并矩阵，数据预处理
dataframe = pd.read_csv("C:/users/73978/Desktop/bre_20.csv", index_col=0)
featurenames = np.array(dataframe.index)
label = pd.read_csv("C:/users/73978/Desktop/bre_tran.csv")
dataframe = dataframe.T
X, y = dataframe, label
y.index = X.index
dt = X.join(y['x'])
dt.rename(columns={'x':'target'}, inplace = True)

# 核验数据
for i in dt.index:
    if (dt.loc[i].isnull().sum() != 0):
        print('Missing value at ', i)
print('Done!')
dt_features = dt.loc[:,dt.columns!='target']
dt_target = dt.iloc[:,-1]

X_train_all,X_test_all,y_train_all,y_test_all = train_test_split(dt_features,dt_target,test_size=0.20,random_state=0)

#建立Tensorflow 模型
model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train_all.keys())]),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['accuracy'])

model.fit(X_train_all,y_train_all,epochs=1000)
print(model.evaluate(X_test_all,y_test_all))
