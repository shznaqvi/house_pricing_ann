# Facebook | Dataframed Live
# Live Coding: Favorite Techniques of the Experts
# https://www.facebook.com/datacampinc/videos/1963687776989026

# Import Packages

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Does not work in PyCharm
# % matplotlib inline

# this is just to make use of sns styles
sns.set()

# Import data and check out
df = pd.read_csv('./data/covid05sept.csv')

# print(df)
# print(df.head())
# print(df.columns)

# print(df.info())

# print(df.describe())

# print(df.loc[:,[3,6]].head())
# print(df.loc[:,['total_cases','total_deaths' ]])


# print(df.isnull().sum())

# i = df[((df.location == 'World') &( df.Age == 15) & (df.Grade == 'A'))].index
# i = df[(df.total_cases == max(df.total_cases)].index
# df.drop(i)
print(df.iloc[df['total_cases'].argmax(),])

df = df.drop(df[(df.total_cases > 100000)].index)
df = df.drop(df[(df.population > 300000000)].index)
df = df.drop(df[(df.population_density > 2000)].index)
# df = df[df['gdp_per_capita'].notna()]
# df.plot.scatter(x='total_cases', y='population_density')

# plt.show()
# df.plot.scatter(x='mean radius', y='mean texture', c='target')
# sns.lmplot(x='population_density', y='total_deaths_per_million', data=df)

# plt.show()


sns.relplot(x='life_expectancy', y='total_deaths', data=df, hue='population')
# sns.relplot(x='life_expectancy', y='total_deaths_per_million', data=df)

# sns.catplot(y='hospital_beds_per_thousand', x='life_expectancy', data=df, hue='continent')

# sns.pairplot(df_ht)
plt.show()
