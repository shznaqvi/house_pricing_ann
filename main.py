import matplotlib.pyplot as plt
import pandas as pd
from keras import regularizers
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/house_price.csv')

# print(df.head())
# print(sys.path)
dataset = df.values
# print(dataset)

X = dataset[:, 0:10]
Y = dataset[:, 10]

# print(X)

mmScaler = preprocessing.MinMaxScaler()
X_Scale = mmScaler.fit_transform(X)  # all values to be scaled between 0 and 1

X_train, X_test_val, Y_train, Y_test_val = train_test_split(X_Scale, Y, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(X_test_val, Y_test_val, test_size=0.5)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(10,), kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=32, epochs=110, validation_data=(X_val, Y_val))

print(model.evaluate(X_test, Y_test)[1])

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('./figures/houseprice_acc')
plt.show()

model.save('./model/HosusePrice_32x3-ANN.model')
