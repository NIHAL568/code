import tensorflow as tf
import numpy as np
from keras import layers,models
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
plt.figure(figsize=(10,10))
for i in range(9):
plt.subplot(3,3,i+1)
plt.imshow(xtrain[i])
plt.xticks([])
plt.yticks([])
xtrain = xtrain.reshape((60000,28,28,1)).astype('float32')/255.0
xtest = xtest.reshape((10000,28,28,1)).astype('float32')/255.0
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)
model = models.Sequential([
layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
layers.MaxPooling2D((2,2)),
layers.Conv2D(64,(3,3),activation='relu'),
layers.MaxPooling2D((2,2)),
layers.Conv2D(64,(3,3),activation='relu'),
layers.Flatten(),
layers.Dense(64,activation='relu'),
layers.Dense(10,activation='softmax')
])
model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)
print(model.summary())

history =
model.fit(xtrain,ytrain,epochs=15,batch_size=64,validation_split=.2)
test_loss,test_acc = model.evaluate(xtest,ytest)
print(f'test accuracy: {test_acc}')
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
