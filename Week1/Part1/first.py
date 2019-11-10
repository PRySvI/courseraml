import numpy as np
import pandas as pd
import re
from collections import defaultdict

dataframe = pd.read_csv('titanic.csv', index_col='PassengerId')

sexes = dataframe['Sex'].value_counts()
all_ppl = dataframe['Survived'].count()
prcnt = lambda x: round(x / all_ppl * 100, 2) 
survived = dataframe['Survived'].value_counts()[1]
sur_result = prcnt(survived)
pclass = prcnt(dataframe.groupby('Pclass').count()['Survived'][1])

age = dataframe.Age.dropna().mean(), dataframe.Age.dropna().median()

cor= dataframe.SibSp.corr(dataframe.Parch)

rx = re.compile('\W')
s2= dataframe[dataframe.Sex != 'male']['Name']
my_dict = defaultdict(int)
for row in s2:
    for val in re.sub(r'\W+', ' ', row.split('.')[-1]).split(' '):
        my_dict[val]+=1