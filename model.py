import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier


from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import pickle

#load the data
df = pd.read_csv("new_water_potability.csv")
print(df.head())
print(df.tail())

print(df.isnull().sum())
data = df.drop('Unnamed: 0',axis=1)
print(data.columns)

X = data.drop('Potability', axis=1)
y = data.Potability
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=30)
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
print(X_train)

xgb = XGBClassifier(colsample_bytree= 0.7,
 learning_rate = 0.01,
 max_depth = 10,
 min_child_weight =  3,
 n_estimators = 500,
 subsample = 0.5)



xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)


confusion_matrix(y_test,y_pred)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

#pickle the model
pickle.dump(xgb  ,open("model.pkl","wb"))

