### IMPORT STATEMENTS:



import numpy as np

import pandas as pd

from sklearn.linear\_model import LogisticRegression,ElasticNet

from sklearn.model\_selection import train\_test\_split

from sklearn.metrics import mean\_squared\_error, r2\_score,mean\_absolute\_error,root\_mean\_squared\_error,confusion\_matrix,roc\_auc\_score

from sklearn.metrics import f1\_score,accuracy\_score,recall\_score,precision\_score,classification\_report

import os

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder

from sklearn.neighbors import KNeighborsClassifier



import datetime as dt

\-------------------------------------------------------------------------------------------------------------

DF = pd.read\_csv(r"../Datasets/sonar.csv")

DF



\----------------------------------------------------------------------------------------------------------------

### ENCODING:

le=LabelEncoder() 

hr\['left']=le.fit\_transform(hr\['left'])

X = hr.drop('left',axis=1)

y=hr\['left']

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.3, random\_state=26, stratify=hr\['left'])



\-------------------------------------------------------------------------------------------------------------------

### Onehotencoder



from sklearn.compose import make\_column\_selector,ColumnTransformer



ohe=OneHotEncoder(sparse\_output=False,drop="first").set\_output(transform="pandas")

trans = ColumnTransformer(

&#x20;   transformers=\[("OHE", ohe,make\_column\_selector(dtype\_include=object))],remainder="passthrough",

&#x20;   verbose\_feature\_names\_out=False).set\_output(transform="pandas")



X\_trn\_ohe = trans.fit\_transform(X\_train)

X\_tst\_ohe = trans.transform(X\_test)



##### usingKNN

from sklearn.neighbors import KNeighborsClassifier

ks=\[1,2,3,4,5,6,7,8,9,10]

scores = \[]

for k in ks:

&#x20;   knn= KNeighborsClassifier(n\_neighbors=k)

&#x20;   knn.fit(X\_trn\_ohe, y\_train)

&#x20;   y\_pred\_prob = knn.predict\_proba(X\_tst\_ohe)

&#x20;   scores.append(\[k, roc\_auc\_score(y\_test, y\_pred\_prob\[:,1])])

df\_scores = pd.DataFrame(scores, columns=\['ks', 'score'])

df\_scores.sort\_values('score', ascending=False)

\-----------------------------------------------------------------------------------------------------------------

solvers = \['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']

Cs = np.linspace(0.001, 5, 20)

scores = \[]

for s in solvers:

&#x20;   for c in Cs:

&#x20;       lr=LogisticRegression(solver=s, C=c, max\_iter=100)  # max\_iter=100 by default sometimes ignore it bcz it may overfit

&#x20;       lr.fit(X\_trn\_ohe, y\_train)

&#x20;       y\_pred = lr.predict(X\_tst\_ohe)

&#x20;       scores.append(\[s, c, f1\_score(y\_test, y\_pred, pos\_label=1)])  # pos\_label by default 1, given because we want score for 1 only not 0



