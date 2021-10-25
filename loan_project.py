import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_info = pd.read_csv('lending_club_info.csv', index_col = 'LoanStatNew')
print(data_info.loc['dti']['Description'])


df = pd.read_csv('lending_club_loan_two.csv')
df.info()
df.isnull().sum()
len(df)

#Data analysis-------------------------------------
sns.countplot(data = df, x ='loan_status' )

plt.figure(figsize = (10,6))
sns.displot(data = df, x = 'loan_amnt', bins = 40)
plt.xlim(0,45000)

cor = df.corr()

plt.figure(figsize = (10,5), dpi = 200)
sns.heatmap(cor,cmap='plasma', annot = True)

plt.figure(figsize = (10,5), dpi = 200)
sns.scatterplot(data = df, x = 'installment', y = 'loan_amnt')

plt.figure(figsize = (10,5), dpi = 200)
sns.boxplot(data = df, x = 'loan_status', y = 'loan_amnt')

df.groupby(['loan_status']).describe()['loan_amnt']

df['grade'].value_counts()
df['sub_grade'].value_counts()

plt.figure(figsize = (10,5), dpi = 200)
sns.countplot(data = df, x = 'grade', hue ='loan_status' )

plt.figure(figsize = (10,5), dpi = 200)
sns.countplot(data = df, x = sorted(df['sub_grade']), palette = 'plasma', hue='loan_status' )


last_stat = df[(df['grade']=='F')|(df['grade']=='G')]

plt.figure(figsize = (10,5), dpi = 200)
sbgrade_order = sorted(last_stat['sub_grade'].unique())
sns.countplot(data = last_stat,x = 'sub_grade', hue = 'loan_status', order = sbgrade_order, palette = 'plasma')

#Data engineering-----------------------------------------
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
correlation = df.corr()['loan_repaid']
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot.bar()

df['emp_title'].value_counts()
df['emp_title'].nunique()
df = df.drop('emp_title',axis = 1)

df['emp_length'].unique()
#emp_order = sorted(df['emp_length'].dropna().unique())

emp_length_order = [ '< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years', '10+ years']

plt.figure(figsize = (12,5), dpi=200)
sns.countplot(data = df, x = 'emp_length', order = emp_length_order)

plt.figure(figsize = (12,5), dpi=200)
sns.countplot(data = df, x = 'emp_length', order = emp_length_order, hue = 'loan_status')

emp_charged = df[df['loan_status']=='Charged Off'].dropna().groupby(['emp_length']).count()['loan_status']
emp_full = df[df['loan_status']=='Fully Paid'].dropna().groupby(['emp_length']).count()['loan_status']
emp_result = emp_charged/emp_full
emp_result.plot(kind = 'bar')
df = df.drop('emp_length',axis = 1)

df = df.drop('title', axis = 1)

df['mort_acc'].value_counts()
df.corr()['mort_acc'].sort_values()
acc_avg = df.groupby('total_acc').mean()['mort_acc']

def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return acc_avg[total_acc]
    else:
        return mort_acc
    
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']),axis = 1)
  
df = df.dropna()  
df.select_dtypes('object').columns
df['term'] = df['term'].apply(lambda term: int(term[:3]) )
df = df.drop('grade', axis = 1)

dummies = pd.get_dummies(df['sub_grade'], drop_first = True)
df = pd.concat([df.drop('sub_grade', axis=1), dummies], axis = 1)

dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']], drop_first = True)
df = pd.concat([df.drop(['verification_status','application_type', 'initial_list_status','purpose'], axis=1), dummies], axis = 1)
df['home_ownership'].value_counts()
    

df = pd.concat([df.drop('home_ownership', axis=1), dummies], axis = 1)
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

df['zip_code'] = df['address'].apply(lambda address:address[-5:])
dummies = pd.get_dummies(df['zip_code'], drop_first = True)
df = pd.concat([df.drop(['zip_code','address'], axis=1), dummies], axis = 1)

df = df.drop('issue_d', axis = 1)
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
df['earliest_cr_line'] = df.earliest_cr_line.apply(lambda earliest_cr_line: earliest_cr_line.year)
df = df.drop('loan_status',axis = 1)

X = df.drop('loan_repaid', axis = 1).values
y = df['loan_repaid'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.75, test_size = 0.25)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience = 25)
model = Sequential()

model.add(Dense(78, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X_train,y_train, epochs = 600,validation_data = (X_test, y_test),callbacks=[early_stop])
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

predictions = model.predict_classes(X_test)
print(accuracy_score(y_test, predictions))
print(classification_report(predictions,y_test))
print(confusion_matrix(y_test, predictions))

from tensorflow.keras.models import load_model
model.save('full_data_project_model.h5')  


