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

#def root_mean_squared_error(y_true, y_pred):
#    return K.sqrt(K.mean(K.square(y_pred - y_true)))

#keras.utils.get_custom_objects().update({'root_mean_squared_error': root_mean_squared_error})


model = keras.models.load_model("Housing_MLP_5")

model.evaluate(X_test, y_test)

#model.predict(X_test[:1,:])

print(X_test.iloc[:1,:])

print(y_test.iloc[:1,:])

#print(model.predict(X_test.iloc[:1,:]))

mean = 206581.870283
std =  115096.397139

predict = model.predict(X_test.iloc[:1,:])

predict =  predict * std
print(predict)

predict = predict + mean
print(predict)

print((0.3208 * std) + mean)
print((2.137496 * std) + mean)




