import tensorflow as tf
from keras import models,layers
from keras.datasets import cifar10
from keras.utils import to_categorical
(xtrain,ytrain),(xtest,ytest)=cifar10.load_data()
xtrain,xtest = xtrain/255.0,xtest/255.0
ytrain = to_categorical(ytrain,10)
ytest = to_categorical(ytest,10)
print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)
model = models.Sequential()
model.add(layers.Flatten(input_shape=(32,32,3)))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)
model.summary()
history =
model.fit(xtrain,ytrain,epochs=15,validation_data=(xtest,ytest))
model.evaluate(xtest,ytest)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
