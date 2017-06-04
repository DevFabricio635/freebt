import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
import gc
import os

model = load_model("freebitcoin_model3.h5")

dados = pd.read_csv("file.csv", sep=',')

x = dados.ix[:, 0:65]
y = np.ravel(dados.win)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# model = Sequential()
# model.add(Dense(24, activation='relu', input_shape=(65,)))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4000, batch_size=1000, verbose=2)

model.save("freebitcoin_model3.h5")

# y_pred = model.predict(x_test)
y_pred = model.predict(x_train)

print("EM TESTE")
print(y_pred[:10])
# print(y_test[:10])
print(y_train[:10])

# score = model.evaluate(x_test, y_test, verbose=1)
score = model.evaluate(x_train, y_train, verbose=1)

print(score)

gc.collect()
