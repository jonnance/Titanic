# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:59:00 2017

@author: v587478
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')

train = pd.read_csv('H:/GIM Strategy and Execution/Knowledge/Data and Python/titanic/train.csv')
del(train['Ticket'])

del(train['PassengerId'])

train[['Pclass', 'Survived']].groupby(['Pclass']).mean()
train[['Pclass', 'Survived']].plot()

sns.countplot( x= 'Pclass', data = train)
sns.countplot(x = 'Survived', hue = 'Embarked', data = train, order = [1,0])

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

#Plot to see % of people from each embarkment survived
embarkedmean = train[['Survived', 'Embarked']].groupby(['Embarked'], as_index = False).mean()
sns.barplot(x = 'Embarked', y = 'Survived', data = embarkedmean, ax = axis2)

#Plot to see % of male / female survived
sexmean = train[['Survived', 'Sex']].groupby(['Sex'], as_index = False).mean()
sns.barplot('Sex', 'Survived', data = sexmean, ax = axis1)

#plot scatter of age vs fare w/ survival as colors
plt.scatter(train['Age'], train['Fare'], c= train['Survived'], cmap=plt.cm.coolwarm)

tMale = train.loc[train['Sex']=='male']
plt.scatter(tMale['Age'], tMale['Fare'], c= tMale['Survived'], cmap=plt.cm.coolwarm)
