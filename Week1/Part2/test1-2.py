# -*- coding: utf-8 -*-

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

dataset = pd.read_csv('titanic.csv', index_col='PassengerId')

dataset1 = dataset[['Pclass','Fare','Age','Survived','Sex']]
dataset1.Sex = dataset1.Sex.apply(lambda x: 1 if x=="male" else 0)

dataset1 = dataset1.dropna()
survived = dataset1.Survived
dataset1.drop('Survived', axis=1, inplace=True)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(dataset1, survived)
importances = clf.feature_importances_

