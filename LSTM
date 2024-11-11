print(datetime.now(pytz.timezone('Asia/Kolkata')))
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU
(xtr, ytr), (xte, yte) = imdb.load_data(num_words=10000)
xtr = pad_sequences(xtr, maxlen=200)
xte = pad_sequences(xte, maxlen=200)
l_model = Sequential([
    Embedding(input_dim=10000, output_dim=128,input_length=200),
    LSTM(units=128),
    Dense(1, activation='sigmoid')
    ])
l_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
l_history = l_model.fit(xtr, ytr, epochs=5, batch_size=64, validation_split=0.2)
loss, acc = l_model.evaluate(xte, yte)
print('Test Accuracy of LSTM:', round(acc * 100, 2))
g_model = Sequential([
    Embedding(input_dim=10000, output_dim=128,input_length=200),
    GRU(units=128),
    Dense(1, activation='sigmoid')
    ])
g_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
g_history = g_model.fit(xtr, ytr, epochs=5, batch_size=64, validation_split=0.2)
loss, acc = g_model.evaluate(xte, yte)
print('Test Accuracy of GRU:', round(acc * 100, 2))
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(l_history.history['accuracy'],label='LSTM(Train)')
plt.plot(l_history.history['val_accuracy'],label='LSTM(Validation)')
plt.plot(g_history.history['accuracy'],label='GRU(Train)')
plt.plot(g_history.history['val_accuracy'],label='GRU(Validation)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(l_history.history['loss'],label='LSTM(Train)')
plt.plot(l_history.history['val_loss'],label='LSTM(Validation)')
plt.plot(g_history.history['loss'],label='GRU(Train)')
plt.plot(g_history.history['val_loss'],label='GRU(Validation)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
