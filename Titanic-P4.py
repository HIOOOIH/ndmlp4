# -*- coding: UTF-8 -*-
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据并设置为df
csv = pd.read_csv('Titanic.csv')
data_train = pd.DataFrame(csv)

# 删除缺失数据
pd.DataFrame.dropna(data_train)

data_train

# 数据：乘客id，存活是1，舱位1等2等3等，名字，性别，年龄，兄弟姐妹或配偶，父母或孩子，票号，票价，船舱号，上船地
# PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked

# 对存活率来说，乘客id、名字、票号相关性不大，同时船舱号缺失的太多了，所以也不分析
# 分析舱位等级、性别、年龄、上船地、家庭成员，票价跟舱位等级相关联，所以只分析了舱位等级。


'''
从舱位等级可以看出一等舱存活率最高，其次是二等舱，最次是三等舱。
'''

'''
取出某几列 https://jingyan.baidu.com/article/f96699bbf6fc95894e3c1bab.html

groupby的用法 http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
'''
pclass_survived = data_train[['Pclass', 'Survived']].groupby(['Pclass']).mean()


'''
从性别来看，女性比男性活下来的更多。
'''
sex_survived = data_train[['Sex', 'Survived']].groupby(['Sex']).mean()

'''
年龄这个不知道怎么解决
'''
age_survived = data_train[['Age', 'Survived']].groupby(['Age']).mean()


'''
从上船地来看,从C口上船的人活下来的最多，其次是Q，再次是S。
'''

embarked_survived = data_train[['Embarked', 'Survived']].groupby(['Embarked']).mean()


'''
从家庭成员来看，这个结果是说4个人的家庭存活率最高？
'''
# 算上本身
data_train['FamilyMember'] = data_train['SibSp'] + data_train['Parch'] + 1

family_survived = data_train[['FamilyMember', 'Survived']].groupby(['FamilyMember']).mean()






