# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Find the null and duplicate values.
3. Using logistic regression find the predicted values of accuracy and confusion matrices.
4. Display the results.
## Program:
```
/*
Program to implement the Logistic Regression Model to Predict the Placement Status of Students.
Developed by: SREEKUMAR S
RegisterNumber:212223240157
**
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
## 1. Placement Data
![Screenshot 2024-04-05 142443](https://github.com/guru14789/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151705853/e24d3c17-1161-4cb0-84cb-7ea09e6e2093)

## 2.Salary Data
![Screenshot 2024-04-05 142453](https://github.com/guru14789/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151705853/47d9f1f4-86af-45f0-b091-17f2193cdb36)

## 3.Checking the null function()
![Screenshot 2024-04-05 142453](https://github.com/guru14789/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151705853/8310cdc7-cab2-4f6a-9d0d-c596c3547b5a)

## 4.Data Duplicate
![Screenshot 2024-04-05 142508](https://github.com/guru14789/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151705853/201429d4-879f-4d72-ac43-94a17a609448)

## 5.Print Data
![Screenshot 2024-04-05 142637](https://github.com/guru14789/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151705853/a3da5865-2072-4a19-8ce9-c9c7aafa2449)

## 6.Data Status
![Screenshot 2024-04-05 142655](https://github.com/guru14789/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151705853/bbdc8b75-9a6a-4768-bdd2-4d820f4e34a9)

## 7.y_prediction array
![ml exp 4 ei](https://github.com/Rama-Lekshmi/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118541549/6d21b45c-3716-42ea-bfd8-0e6d4bd96875)
## 8.Accuracy value
![238188899-f13d398f-7b71-49bd-b787-2ae8b6c86f49](https://github.com/guru14789/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151705853/b6ac3f97-29ae-4ec0-afd2-730f1fa8848a)

## 9.Confusion matrix
![238188940-bea05d71-7a26-41df-b4af-84dc2d0788f6](https://github.com/guru14789/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151705853/a60d6df0-24a2-4f5f-bb3a-fef2c0f2b0c8)

## 10.Classification Report
![238188964-495dae98-1d9d-4fcf-bb3c-20e259485d07](https://github.com/guru14789/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151705853/6c35115e-de7c-4494-9220-be0f804bde18)


## 11.Prediction of LR
![238189054-17ef5d1d-79dc-41cb-a9d1-b05eff1c61dd](https://github.com/guru14789/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/151705853/d85e8348-aea6-49e9-b069-0bfbf6b9cd89)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
