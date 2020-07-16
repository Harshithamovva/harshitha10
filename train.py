import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

def read_data_pandas():
    data_frame=pd.read_csv('train.csv')
    #X=data_frame[['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']]
    X=data_frame[['Sex','SibSp','Parch','Fare']]
    #X=data_frame.iloc[:,[4,6,7,9]] 
    #X=X.replace('', nan)
    # print(X.dtypes)
    # print(X.describe())
    # print(X.isna().sum())
    # print(data_frame.corr())
    y=data_frame.iloc[:,1]
    # mms = MinMaxScaler()
    # rs = RobustScaler()
    # X=mms.fit_transform(X)
    # print(X) 
    # imputer=SimpleImputer(missing_values='NaN',strategy='mean')
    # imputer = IterativeImputer(random_state=0)
    # X=imputer.fit_transform(X)
    # print(X)
    return X,y

def apply_ML(X,y):
    X=np.array(X)
    y=np.array(y)
    X_train,X_val,Y_train,Y_val = train_test_split(X,y,test_size=0.2,random_state=42)
    clf = DecisionTreeClassifier(max_depth=6,min_samples_leaf=8)
    clf.fit(X_train,Y_train)
    export_graphviz(clf,out_file='titanic.dot',filled=True,feature_names=['Glucose','BMI','fare'])
    print("TR score",clf.score(X_train,Y_train))
    print("vali score",clf.score(X_val,Y_val))
    # X_test=np.array([93,39]).reshape(1,-1)    
    # yvalue=clf.predict_proba(X_val)
    yvalue=clf.predict(X_val)
    # print(yvalue)
    # class_1_probas = yvalue[:,1]
    # y_predict=clf.predict_proba(X_test)
    # print(y_predict)
    cm = confusion_matrix(Y_val,yvalue)
    print(cm)
    
if __name__ == "__main__":
    X,y=read_data_pandas()
    # visualize(X,Y)
    apply_ML(X,y)
    
    