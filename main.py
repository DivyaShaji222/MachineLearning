import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mlxtend.preprocessing import TransactionEncoder

from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,fbeta_score

dataset = pd.read_csv('Train_Dataset.csv')
dataset = dataset.dropna()
x = dataset.drop(columns=['Credit_Bureau','Social_Circle_Default','Age_Days','Employed_Days','Score_Source_1','Score_Source_2','Default','Accompany_Client','Client_Income_Type','Client_Education','Client_Gender','Client_Housing_Type','Client_Occupation','Client_Permanent_Match_Tag','Type_Organization','Client_Marital_Status','Loan_Contract_Type','Client_Contact_Work_Tag'])
y=dataset['Default']
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=.25,random_state=0)
# print(x_train.shape,x_test.shape)

classiifier = RandomForestClassifier(n_estimators=200,random_state=42)
classiifier.fit(x_train,y_train)
print(classiifier.score(x_train,y_train))
print(classiifier.score(x_test,y_test))

with open('model345.pkl', 'wb') as file:
    pickle.dump(classiifier, file)