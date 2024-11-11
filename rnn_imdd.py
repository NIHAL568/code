import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense
num_words=10000
max_length=200
(xtr,ytr),(xte,yte)=imdb.load_data(num_words=num_words)
xtr,xte=pad_sequences(xtr,maxlen=max_length),pad_sequences(xte,maxlen=max_length)
model=Sequential([
    Embedding(input_dim=num_words,output_dim=128,input_length=max_length),
    LSTM(128),
    Dense(1,activation='sigmoid')
    ])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(xtr,ytr,validation_split=0.2,epochs=50,batch_size=64)
loss,acc=model.evaluate(xte,yte)
print("Test accuracy:",round(acc*100,4))
test_seq=np.reshape(xte[1],(1,-1))
pred=model.predict(test_seq)[0]
if round(pred[0])==1:
  print('Positive Review')
else:
  print('Negative Review')
print(yte[1])
