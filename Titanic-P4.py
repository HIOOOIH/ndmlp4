# -*- coding: UTF-8 -*-
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据并设置为df
csv = pd.read_csv('Titanic.csv')
titanic_df = pd.DataFrame(csv)
titanic_df.info()

# 1. 提出问题
# 详见ipynb


# 2. 数据清理
changed_df = titanic_df.copy()
changed_df['Embarked'] = changed_df['Embarked'].dropna()

# http://pandas.pydata.org/pandas-docs/stable/missing_data.html
changed_df = titanic_df.fillna(titanic_df.mean()['Age'])


# 3. 探索分析
def factor_explore(factor):
    factor_survived = changed_df[[factor, 'Survived']].groupby([factor]).mean()
    factor_survived.plot(kind='bar')
    plt.title(factor + ' VS Survial Rate')
    plt.ylabel(factor + ' Rate')
    
# 舱位等级
factor_explore('Pclass')

# 性别
factor_explore('Sex')

# TODO 港口 
factor_explore('Embarked')

# 家庭成员
changed_df['FamilyMember'] = changed_df['SibSp'] + changed_df['Parch'] + 1

factor_explore('FamilyMember')

# 年龄
changed_df['Age'].hist()
plt.title('Age Distribution')
plt.ylabel('Frequency')
plt.xlabel('Age')


# 补充
# 性别的比例
sex_count = changed_df[['Sex','Survived']].groupby(['Sex']).size()
sex_count

# 幸存者与死亡者的比例
Survived_count =  changed_df.groupby('Survived')['Survived'].count()
print (Survived_count)
plt.pie(Survived_count, labels=['Non-survived','Survived'], autopct='%.1f%%')

# 4. 结论总结
# 详见ipynb