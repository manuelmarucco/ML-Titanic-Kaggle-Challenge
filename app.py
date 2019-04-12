# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:21:17 2019

@author: Manue
"""

## IMPORT LIBRARIES

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer # Missing values
from sklearn.ensemble import RandomForestClassifier # ML Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder


#%% SET DATA

df_raw = pd.read_csv('data/train.csv')
df = pd.read_csv('data/train.csv')

print(df.info())

""" DATA INFORMATION
Variable	Definition	Key
survival 	Survival 	0 = No, 1 = Yes
pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd
sex 	Sex
Age 	Age in years
sibsp 	# of siblings / spouses aboard the Titanic
parch 	# of parents / children aboard the Titanic
ticket 	Ticket number
fare 	Passenger fare
cabin 	Cabin number
embarked 	Port of Embarkation    C = Cherbourg, Q = Queenstown,
                                   S = Southampton
"""
""" DATA TYPES
PassengerId      int64      dropped
Survived         int64      Categorical
Pclass           int64      Categorical
Name            object      dropped
Sex             object      Categorical
Age            float64      Ordinal
SibSp            int64      Ordinal
Parch            int64      Ordinal
Ticket          object      Categorical
Fare           float64      Ordinal
Cabin           object      Categorical
Embarked        object      Categorical


NOTES
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
"""

#%% DATA PREPARATION

print("DATASET: MISSING VALUES %")
print(df.isnull()
        .sum().apply(lambda x: str('%.1f'%(x/len(df.index)*100)) + ' %'))
"""
NOTES: DATASET MISSING VALUES

PassengerId     0.0 %
Survived        0.0 %
Pclass          0.0 %
Name            0.0 %
Sex             0.0 %
Age            19.9 %
SibSp           0.0 %
Parch           0.0 %
Ticket          0.0 %
Fare            0.0 %
Cabin          77.1 %
Embarked        0.2 %

"""

"""
REMOVE FEATURES
"""
df.drop(columns='PassengerId', inplace = True)
df.drop(columns='Name', inplace = True)

"""
MISSING VALUES
"""
impOrdinal = SimpleImputer(strategy='mean')
impNominal = SimpleImputer(strategy='most_frequent')
impNumerical = SimpleImputer(strategy='constant', fill_value=0)

# CATEGORICAL FEATURES
df['Pclass'] = impNominal.fit_transform(df[['Pclass']])

for col in df.select_dtypes(include=['object']):
    df[col] = impNominal.fit_transform(df[[col]])


# ORDINAL FEATURES
df['SibSp'] = impNumerical.fit_transform(df[['SibSp']])
df['Parch'] = impNumerical.fit_transform(df[['Parch']])

for col in df.select_dtypes(include=['float64']):
    df[col] = impNominal.fit_transform(df[[col]])

"""
NOISY VALUES
"""
df['Age'] = df['Age'].apply(lambda x: int(np.ceil(x)))


"""
ENCODING CATEGORICAL FEATURES TO NUMERICAL FEATURES
"""
encoder = LabelEncoder()

for col in df.select_dtypes(include=['object']):
    df[col] = encoder.fit_transform(df[col])



#%% TRAINING MODEL
train_features = df.iloc[:,1:len(df.columns)]
train_targets = df.iloc[:,0]

"""
Train the model
"""

tree = RandomForestClassifier(criterion = 'gini', n_estimators=1000,
                              max_features='sqrt', oob_score=True,
                              n_jobs = -1).fit(train_features,train_targets)

#%% RESULTS

"""
Check the accuracy
"""

print("\nOOB Accuracy: ",tree.oob_score_*100, '%')
print("\nOOB Error rate: ",100-tree.oob_score_*100, '%')


"""
Caclulate Feature Importance by permutation
"""
feature_importance_table = pd.DataFrame(
                                    {'feature': list(train_features.columns),
                                     'importance': tree.feature_importances_})

print(feature_importance_table.sort_values('importance', ascending = False))


#%% FEATURE IMPORTANCE VERYFICATION

print("\n MUCH MORE FEMALES SURVIVED THAN MALE")

survived_females = len(df_raw[df_raw.Sex == 'female'][df_raw['Survived'] == 1])
total_females = len(df_raw[df_raw.Sex == 'female'])
print('Female Survival Rate',survived_females/total_females*100,'%')


survived_males = len(df_raw[df_raw.Sex == 'male'][df_raw['Survived'] == 1])
total_males = len(df_raw[df_raw.Sex == 'male'])
print('Male Survival Rate',survived_males/total_males*100,'%')

## TODO: Replace the female calculation with the bottom IMPORTANCE
"""


#Discrete Variable Correlation by Survival using
#group by aka pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
for x in data1_x:
    if data1[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')

"""
