import tensorflow as tf
from tensorflow.keras.applications import VGG19
import numpy as np
from keras import layers,models
from keras.datasets import mnist
from keras.utils import to_categorical
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
Xtrain.shape,xtest.shape
xtrain =
np.repeat(tf.image.resize(xtrain[...,np.newaxis],(32,32)).numpy(),3,ax
is=-1)
xtest =
np.repeat(tf.image.resize(xtest[...,np.newaxis],(32,32)).numpy(),3,axi
s=-1)
xtrain = xtrain/255.0
xtest = xtest/255.0
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)
base_model = VGG19(
include_top=False,
weights = 'imagenet',
input_shape=(32,32,3)
)
base_model.trainable = False
model = models.Sequential([
base_model,
layers.Flatten(),
layers.Dense(256,activation='relu'),
layers.Dense(10,activation='softmax')
])
model.compile(
optimizer='adam',

loss='categorical_crossentropy',
metrics=['accuracy']
)
history =
model.fit(xtrain,ytrain,epochs=3,batch_size=64,validation_split=.2)
test_loss,test_acc = model.evaluate(xtest,ytest)
print(test_loss,test_acc)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
