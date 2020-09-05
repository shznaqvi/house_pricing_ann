# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

sns.set()

# Import data
df = pd.read_csv('data/cancer.csv')

# Get the feel of data
print(df.head())
wait = input("PRESS ENTER TO CONTINUE.")
print(df.info())
wait = input("PRESS ENTER TO CONTINUE.")

print(df.describe())
wait = input("PRESS ENTER TO CONTINUE.")

print(df.shape)
wait = input("PRESS ENTER TO CONTINUE.")

# df.plot.scatter(x='mean radius', y='mean texture', c='target')
# plt.show()

# sns.lmplot(x='mean radius', y='mean texture', data=df)
# plt.show()

# sns.lmplot(x='mean radius', y='mean texture', hue='target', data=df)
# plt.show()

# df.plot.scatter(x='mean radius', y='mean perimeter', c='target')
# plt.show()

# Creating subset of data
df_sub1 = df

# sns.pairplot(hue='target', data=df_sub1, height=3)
# plt.show()

X = df_sub1.drop('target', axis=1).values
Y = df_sub1['target'].values

mmScaler = preprocessing.MinMaxScaler()
X_Scale = mmScaler.fit_transform(X)

X_train, X_test_val, Y_train, Y_test_val = train_test_split(X_Scale, Y, test_size=0.2, random_state=42, stratify=Y)
X_test, X_val, Y_test, Y_val = train_test_split(X_test_val, Y_test_val, test_size=0.5)

# Using Decision Tree
# clf = tree.DecisionTreeClassifier(max_depth=2)
# print(clf.fit(X_train, Y_train))
# print(clf.score(X_test, Y_test))
# print(clf.score(X_val, Y_val))

# Using ANN
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(30,), kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.33))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))

print(model.evaluate(X_test, Y_test)[1])

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper right')
plt.show()
