import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('kc_house_data.csv')
df.isnull().sum()
df.describe().transpose()
df = df.sort_values('price', ascending = False).iloc[216:]

plt.figure(dpi = 200, figsize=(10,6))
sns.distplot(df['price'])

sns.countplot(df['bedrooms'])

plt.figure(dpi = 200, figsize=(10,6))
sns.scatterplot(x='price', y='sqft_living', data = df)

df.corr()['price'].sort_values()

non_top_1_perc = df.sort_values('price', ascending = False).iloc[216:]

plt.figure(figsize = (10,6), dpi=200)
sns.scatterplot(x='long',y='lat', data = non_top_1_perc,edgecolor = None, alpha = 0.5, hue = 'price', palette='plasma' )

df.head()
df = df.drop('id', axis = 1)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df.date.apply(lambda date: date.year)
df['month'] = df.date.apply(lambda date: date.month)

plt.figure(figsize = (10,6), dpi=200)
sns.boxplot(x = 'month', y = 'price', data = df)

df.groupby('month').mean()['price']
df = df.drop('date',axis = 1)
df['zipcode'].values_counts()
df = df.drop('zipcode',axis = 1)


X = df.drop('price', axis = 1).values
y = df['price'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, train_size=0.7)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x = X_train, y = y_train, epochs=300,validation_data = (X_test, y_test), batch_size = 128)

model.history.history
losses = pd.DataFrame(model.history.history)
losses.plot()

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
predictions = model.predict(X_test)
mean_absolute_error(y_test, predictions)
np.sqrt(mean_squared_error(y_test, predictions))
explained_variance_score(y_test, predictions)
