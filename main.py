import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/hospital_admissions_2003.csv')

print(df.shape)

# print(df.head())
# print(sys.path)
# wait = input("PRESS ENTER TO CONTINUE.")

dataset = df.values
# print(dataset)
# wait = input("PRESS ENTER TO CONTINUE.")

X = dataset[:, 0:10]
Y = dataset[:, 10]

# print(X)
# wait = input("PRESS ENTER TO CONTINUE.")

mmScaler = preprocessing.MinMaxScaler()
X_Scale = mmScaler.fit_transform(X)  # all values to be scaled between 0 and 1
# print(X_Scale)
# wait = input("PRESS ENTER TO CONTINUE.")

X_train, X_test_val, Y_train, Y_test_val = train_test_split(X_Scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_test_val, Y_test_val, test_size=0.5)

# print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
# wait = input("PRESS ENTER TO CONTINUE.")

# print(len(X_train))
# wait = input("PRESS ENTER TO CONTINUE.")


model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(10,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Addressing overfitting
# model.add(Dense(32, activation='relu', input_shape=(10,), kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dropout(0.3))
# model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation='sigmoid'))

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))

train_acc = model.evaluate(X_train, Y_train)[1]
test_acc = model.evaluate(X_test, Y_test)[1]

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
# plt.savefig('./figures/houseprice_acc')
plt.show()

# Checking overfitting (training accuracy should be lower than testing accuracy)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='test')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
# plt.savefig('./figures/houseprice_acc')
plt.show()

model.save('./model/HosusePrice_32x3-ANN.model')
