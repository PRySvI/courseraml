import numpy as np
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier

feautures = []
feautures_names = []
with open('files/wine.data') as file:
    lines = file.readlines()
    for row in lines:
        splitedArr = [float(s) for s in row.split(',')]
        feautures_names.append(int(splitedArr[0]))
        feautures.append(splitedArr[1:])

from sklearn.model_selection import KFold

classifer = KNeighborsClassifier(n_neighbors=10)
print(feautures)
print(feautures_names)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
estimations = []
for train, test in kf.split(feautures):
    xtest = np.array(feautures)[test.astype(int)]
    xtrain = np.array(feautures)[train.astype(int)]
    xtrain_n = np.array(feautures_names)[train.astype(int)]
   # print("%s %s" % (xtest, xtrain))
    '''
    print('*-' * 25)
    print(train)
    print('te-' * 25)
    print(test)
    print('tr-'*25)
    '''
    classifer.fit(xtrain, xtrain_n)
    pred = classifer.predict(xtest)
    cureman  =  np.mean(pred == np.array(feautures_names)[test.astype(int)])
    estimations.append(cureman)

print(np.mean(estimations))

