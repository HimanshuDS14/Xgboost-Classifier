import pandas as pd

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report

import xgboost



data = pd.read_csv("modeling.csv")
print(data.head(10))
print(data.shape)
print(data.info())



x = data.iloc[:,3:13].values
y = data.iloc[:,13].values

print(x)
print(y)

le1 = LabelEncoder()
le2 = LabelEncoder()

x[:,1] = le1.fit_transform(x[:,1])
x[:,2] = le2.fit_transform(x[:,2])

OHE = OneHotEncoder(categorical_features=[1])
x = OHE.fit_transform(x).toarray()

x = x[:,1:]
print(x)

train_x , test_x , train_y ,test_y = train_test_split(x,y,test_size=0.2 , random_state=0)


classifier = xgboost.XGBClassifier()
classifier.fit(train_x  , train_y)

pred =classifier.predict(test_x)

print(confusion_matrix(test_y , pred))
print(classification_report(test_y , pred))




