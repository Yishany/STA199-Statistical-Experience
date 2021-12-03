#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:02:31 2021

@author: christinayu
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pandas import Series,DataFrame

plt.rc('font',family='Cambria',size=13)

os.getcwd()
os.chdir('/Users/christinayu/Desktop/Python/AI人工智能/')


data_train=pd.read_excel('Tanic1.xlsx')
data_train.columns
data_train.info()


data_train.describe()

import matplotlib.pyplot as plt
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
plt.rc('font', family='SimHei', size=6)   #作图显示中文
data_train['Survived']

plt.subplot(2,3,1)
data_train['Survived'].value_counts(dropna=False).plot(kind='bar')
plt.title(u'获救情况(1为获救)')
plt.ylabel(u"人数")



plt.subplot((2,3,2)
data_train['Pclass'].value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.subplot((2,3,3)
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")                         # sets the y axis lable
plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
plt.title(u"按年龄看获救分布 (1为获救)")

plt.subplot2grid((2,3),(1,0), colspan=2)
data_train[data_train['Pclass'] == 1]['Age'].plot(kind='kde')   #画出来一个曲线的分布图
data_train[data_train['Pclass'] == 2]['Age'].plot(kind='kde')
data_train.Age[data_train['Pclass'] == 3].plot(kind='kde') #这个写法和上面的是一样的
plt.xlabel(u"年龄")# plots an axis lable
plt.ylabel(u"密度") 
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")  
plt.show()



#看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train[data_train['Survived'] == 0]['Pclass'].value_counts()
Survived_1 = data_train[data_train['Survived'] == 1]['Pclass'].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级") 
plt.ylabel(u"人数") 

plt.show()


#看看各登录港口的获救情况

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数


Survived_0 = data_train[data_train['Survived'] == 0]['Embarked'].value_counts()
Survived_1 = data_train[data_train['Survived'] == 1]['Embarked'].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口") 
plt.ylabel(u"人数") 

plt.rc('font',family='Cambria',size=13)

plt.show()


#看看各性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_m = data_train[data_train['Sex'] == 'male']['Survived'].value_counts()
Survived_f = data_train[data_train['Sex'] == 'female']['Survived'].value_counts()
df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"按性别看获救情况")
plt.xlabel(u"性别") 
plt.ylabel(u"人数")
plt.show()


#然后我们再来看看各种舱级别情况下各性别的获救情况
fig=plt.figure()
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title(u"根据舱等级和性别的获救情况")

ax1=fig.add_subplot(141)
data_train[data_train['Sex'] == 'female'][data_train.Pclass != 3]['Survived'].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
ax1.legend([u"女性/高级舱"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train[data_train['Sex'] == 'female'][data_train.Pclass == 3]['Survived'].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"女性/低级舱"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train[data_train['Sex'] == 'male'][data_train.Pclass != 3]['Survived'].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/高级舱"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train[data_train['Sex'] == 'male'][data_train.Pclass == 3]['Survived'].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
plt.legend([u"男性/低级舱"], loc='best')

plt.show()

#大家族和兄弟的生存情况
data_train.info()
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['Name'])
df

#父母和孩子的生存情况
g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['Name'])
df


#cabin的值计数太分散了，绝大多数Cabin值只出现一次。感觉上作为类目，加入特征未必会有效
#那我们一起看看这个值的有无，对于survival的分布状况，影响如何吧
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无") 
plt.ylabel(u"人数")
plt.show()

#似乎有cabin记录的乘客survival比例稍高，那先试试把这个值分为两类，有cabin值/无cabin值，一会儿加到类别特征好了


