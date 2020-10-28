#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : pre-test.py
# @Author: Stormzudi
# @Date  : 2020/7/28 18:42

"""
功能：对泰坦尼克号数据进行数据挖掘分析

# 第一部分：数据读取与展示
# 第二部分：特征理解分析
# 第三部分：缺失值填充
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 第一部分：数据读取与展示
# 展示部分的数据
data=pd.read_csv('train.csv')
print(data.head())

# checking for total null values
print(data.isnull().sum())

# 数据报表
print(data.describe())

# 展示获救情况
f, ax = plt.subplots(1, 2, figsize=(6, 4))
data['Survived'].value_counts().plot.pie(explode=[0, 0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot(x ='Survived',data=data,ax=ax[1])
ax[1].set_title('Survived')
# plt.show()


# 第二部分：特征理解分析
# 交叉统计性别和获救之间的关系
var1 = data.groupby(['Sex','Survived'])['Sex'].count()
# 交叉统计等级和获救之间的关系
var2 = pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient(cmap='summer_r').data

# 绘制等级和获救之间的关系图
f,ax=plt.subplots(1,2,figsize=(8,4))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot(x='Pclass', hue='Survived',data=data,ax=ax[1])
# ax[1].set_title('Pclass:Survived vs Dead')
plt.show()

# 分析性别，不同等级下的获救情况
var3 = pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True).style.background_gradient(cmap='summer_r').data
# 不同等级下，获救情况，按年龄分析
sns.factorplot('Pclass','Survived',hue='Sex',data=data)
plt.show()


# 分析 Age 连续值特征对结果的影响
print('Oldest Passenger was of:',data['Age'].max(),'Years')
print('Youngest Passenger was of:',data['Age'].min(),'Years')
print('Average Age on the ship:',data['Age'].mean(),'Years')


# 绘制“小提琴”图，来分析不同等级下，不同年龄短的获救情况。
f,ax=plt.subplots(1,2,figsize=(8,4))
sns.violinplot("Pclass","Age", hue="Survived", data=data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# 第三部分：缺失值填充
data['Initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')  # 匹配一个或多个字母并且以.结尾的所有字符串组合
print(data['Initial'])


