# Import packages
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split

sns.set()

# Import data
df = pd.read_csv('./data/cancer.csv')

# Get the feel of data
# print(df.head())
# print(df.info())
# print(df.describe())

# df.plot.scatter(x='mean radius', y='mean texture', c='target')
# plt.show()

# sns.lmplot(x='mean radius', y='mean texture', data=df)
# plt.show()

# sns.lmplot(x='mean radius', y='mean texture', hue='target', data=df)
# plt.show()

# df.plot.scatter(x='mean radius', y='mean perimeter', c='target')
# plt.show()

# Creating subset of data
df_sub1 = df.iloc[:, [0, 1, 2, 3, -1]]

# sns.pairplot(hue='target', data=df_sub1, height=3)
# plt.show()

X = df.drop('target', axis=1).values
Y = df['target'].values

X_train, X_test_val, Y_train, Y_test_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
X_test, X_val, Y_test, Y_val = train_test_split(X_test_val, Y_test_val, test_size=0.5)

# Using Decision Tree
clf = tree.DecisionTreeClassifier(max_depth=2)
print(clf.fit(X_train, Y_train))
print(clf.score(X_test, Y_test))
print(clf.score(X_val, Y_val))
