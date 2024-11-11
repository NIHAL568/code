import tensorflow as tf
import numpy as np
from keras import models,layers,optimizers
from keras.datasets import cifar10
from keras.utils import to_categorical
(xtrain,ytrain),(xtest,ytest)=cifar10.load_data()
xtrain,xtest = xtrain/255.0,xtest/255.0
ytrain = to_categorical(ytrain,10)
ytest = to_categorical(ytest,10)
print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)

#Xavier
model1 = models.Sequential()
model1.add(layers.Flatten(input_shape=(32,32,3)))
model1.add(layers.Dense(256,activation='relu',kernel_initializer='glorot_uni
form'))
model1.add(layers.Dense(128,activation='relu',kernel_initializer='glorot_uni
form'))
model1.add(layers.Dense(10,activation='softmax',kernel_initializer='glorot_u
niform'))
sgd_optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.9)
model1.compile(
optimizer=sgd_optimizer,
loss = 'categorical_crossentropy',
metrics = ['accuracy']
)
xav = model1.fit(xtrain,ytrain,epochs=25,batch_size=32,validation_split=.2)
xav_score = model1.evaluate(xtest,ytest,batch_size=32)
#Xavier_Dropout
model1_drop = models.Sequential()
model1_drop.add(layers.Flatten(input_shape=(32,32,3)))
model1_drop.add(layers.Dense(256,activation='relu',kernel_initializer='gloro
t_uniform'))
model1_drop.add(layers.Dropout(0.25))
model1_drop.add(layers.Dense(128,activation='relu'))

model1_drop.add(layers.Dense(10,activation='softmax'))
sgd_optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.9)
model1_drop.compile(
optimizer=sgd_optimizer,
loss = 'categorical_crossentropy',
metrics = ['accuracy']
)
xav_drop =
model1_drop.fit(xtrain,ytrain,epochs=25,batch_size=32,validation_split=.2)
xav_drop_score = model1_drop.evaluate(xtest,ytest,batch_size=32)
#Kaiming_He
model2 = models.Sequential()
model2.add(layers.Flatten(input_shape=(32,32,3)))
model2.add(layers.Dense(256,activation='relu',kernel_initializer='he_normal'
))
model2.add(layers.Dense(128,activation='relu',kernel_initializer='he_normal'
))
model2.add(layers.Dense(10,activation='softmax',kernel_initializer='he_norma
l'))
sgd_optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.9)
model2.compile(
optimizer=sgd_optimizer,
loss = 'categorical_crossentropy',
metrics = ['accuracy']
)
he = model2.fit(xtrain,ytrain,epochs=25,batch_size=32,validation_split=.2)
he_score = model2.evaluate(xtest,ytest,batch_size=32)
#Kaiming_He_Dropout
model2_drop = models.Sequential()
model2_drop.add(layers.Flatten(input_shape=(32,32,3)))
model2_drop.add(layers.Dense(256,activation='relu',kernel_initializer='he_no
rmal'))
model2_drop.add(layers.Dropout(0.25))
model2_drop.add(layers.Dense(128,activation='relu'))
model2_drop.add(layers.Dense(10,activation='softmax'))

sgd_optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.9)
model2_drop.compile(
optimizer=sgd_optimizer,
loss = 'categorical_crossentropy',
metrics = ['accuracy']
)
he_drop =
model2_drop.fit(xtrain,ytrain,epochs=25,batch_size=32,validation_split=.2)
he_drop_score = model2_drop.evaluate(xtest,ytest,batch_size=32)

#Batch Normalization
BN = models.Sequential()
BN.add(layers.Flatten(input_shape=(32,32,3)))
BN.add(layers.Dense(256,activation='relu'))
BN.add(layers.BatchNormalization())
BN.add(layers.Activation('relu'))
BN.add(layers.Dense(10,activation='softmax'))
sgd_optimizer = optimizers.SGD(learning_rate=0.01,momentum=0.9)
BN.compile(
optimizer=sgd_optimizer,
loss = 'categorical_crossentropy',
metrics = ['accuracy']
)
BN_hist = BN.fit(xtrain,ytrain,epochs=25,batch_size=32,validation_split=.2)
BN_score = BN.evaluate(xtest,ytest,batch_size=32)
#Results
print('Xavier_score:',xav_score)
print('Xavier_drop_score:',xav_drop_score)
print('He_score:',he_score)
print('He_drop_score:',he_drop_score)
print('Batch Normalization_score:',BN_score)
import matplotlib.pyplot as plt
plt.plot(xav.history['val_accuracy'],label='xavier_initialization')
plt.plot(xav_drop.history['val_accuracy'],label='xavier_dropout')
plt.plot(he.history['val_accuracy'],label='Kaiming_initialization')
plt.plot(he_drop.history['val_accuracy'],label='Kaiming_dropout')
plt.plot(BN_hist.history['val_accuracy'],label='Batch_Normalization')
plt.legend()
plt.show()
