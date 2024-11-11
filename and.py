import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,models,Sequential
from tensorflow.keras.layers import Dense,Activation
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])
model = Sequential()
model.add(Dense(16,input_dim=2,activation='relu',use_bias=False))
model.add(Dense(1,activation='sigmoid',use_bias=False))
model.compile(optimizer='adam',loss='mse')
model.fit(x,y,epochs=500)
l = model.predict(np.array([[1,1]]))
print('prediction for [1,1]:',1 if l>0.5 else 0)

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,models,Sequential
from tensorflow.keras.layers import Dense
x = np.array([3.0])
y = np.array([3.0])
model = Sequential()
model.add(Dense(1,name='D1',input_dim=1,use_bias=False))
model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
model.fit(x,y,epochs=100)
l=model.predict(np.array([15]))
print(l)
