#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 11:27:45 2021

@author: christinayu
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette("Set2")


os.chdir('/Users/christinayu/Desktop/Python/AI人工智能/逻辑回归')
data = pd.read_excel('practice2.xlsx', usecols = [0,1,2,3,4,5,6,7,8,9,10])

data.info()
data.describe()

sns.factorplot(x='Sex',y='Survived',data=data,size=4,aspect=2)

data.pop('Cabin')
data.dropna(axis=0,how='any',inplace=True)
data.pop('Name')
data.pop('Ticket')
data.pop('Embarked')

x = data.iloc[:,1:9]

x.iloc[:,1] = pd.Categorical(x.iloc[:,1]).codes


y = data.iloc[:,0]


print (x.shape)
print (y.shape)



# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train = x
y_train = y
#model = LogisticRegression(C=1e9)
model = LogisticRegression()
model.fit(x_train, y_train)
predict_y = model.predict(x_train)
p = np.mean(predict_y == y_train)
print (p)

print('logistic_model:',model.intercept_,'+x*',model.coef_)

#进行检测
x1=1
x2=2
x3=3
p1=1/(1+np.exp(0.02437491*x1+0.51021416*x2+0.40163878*x3-2.18397006))
print(p1)
#相当于把p1转换成了原来的p,logp/1-p中的p,求得概率
