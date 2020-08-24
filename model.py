import keras
import pandas as pd
from keras import backend as K

training = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")

train = training.iloc[:,1:]
test = test.iloc[:,1:]

mean = train.mean(axis = 0)
train -= mean
std = train.std(axis = 0)
train /= std

test -= mean
test /= std

valid = train.iloc[13500:,:]

X_train = train.iloc[:,:6]
y_train = train.iloc[:,6:]

X_test = test.iloc[:,:6]
y_test = test.iloc[:,6:]

X_valid = valid.iloc[:,:6]
y_valid = valid.iloc[:,6:]


print(X_train.head())
print(y_train.head())

print(X_test.head())
print(y_test.head())

print(X_valid.head())
print(y_valid.head())

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

model = keras.models.Sequential([
    keras.layers.Dense(128,input_shape=[6, ]),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.00000001)
optimizer2 = keras.optimizers.RMSprop()
#model.compile(loss= keras.losses.MeanSquaredError() , optimizer=optimizer)
model.compile(loss= 'mse' , optimizer="rmsprop", metrics=['mae'])
fit_1 = model.fit(X_train, y_train, epochs = 500,
validation_data=(X_valid, y_valid))

model.save("Housing_MLP_5")

import matplotlib.pyplot as plt
loss = fit_1.history['loss']
val_loss = fit_1.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
