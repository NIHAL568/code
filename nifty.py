import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM,SimpleRNN
from tensorflow import keras
from tensorflow.keras import layers,models,Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import
TimeseriesGenerator
data =
pd.read_csv("/home/mits/Documents/nifty.csv",index_col="Date",parse_da
tes=True)[['Open','High','Low','Close']]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(data)
n=int(len(scaled_data)*0.8)
xtrain=scaled_data[:n]
xtest=scaled_data[n:]
n_features = 4
n_length = 3
train_generator = TimeseriesGenerator(xtrain,xtrain,length = n_length)
test_generator = TimeseriesGenerator(xtest,xtest,length = n_length)
model = models.Sequential([
LSTM(50,activation='relu'),
layers.Dense(4)
])
model.compile(
optimizer = 'adam',
loss = 'mean_squared_error',
metrics = ['accuracy']
)
model.fit(train_generator,epochs=50)

print(model.summary())
print('test accuracy',model.evaluate(test_generator)[1])
predictions = model.predict(test_generator)
predictions_real = scaler.inverse_transform(predictions)
test_data_original = scaler.inverse_transform(xtest[n_length:])
import matplotlib.pyplot as plt
variables = ['Open', 'High', 'Low' , 'Close']
for index, value in enumerate(variables):
plt.figure(figsize = (15,3))
plt.plot(data.index[n+n_length:],test_data_original[:,index],label=f'r
eal {value}', color='blue')
plt.plot(data.index[n+n_length:],predictions_real[:,index],
label=f'predicted {value}', color = 'red')
plt.title('NIFTY 50')
plt.xlabel('Date')
plt.ylabel(f'{value} price')
plt.legend()
plt.show()
