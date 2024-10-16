# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder. 
3.Determine test and training data set and apply decison tree regression in dataset. 
4.calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VIJAY R
RegisterNumber: 212223240178 
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
```
![Screenshot 2024-10-16 103327](https://github.com/user-attachments/assets/231b8700-a07b-40c3-b6fb-c15107023f37)

```
data.info()
```
![Screenshot 2024-10-16 103333](https://github.com/user-attachments/assets/4caa80c9-3429-4760-88a9-8a49b1d79ba6)

```
data.isnull().sum()
```
![Screenshot 2024-10-16 103339](https://github.com/user-attachments/assets/fad2901f-6674-4427-8186-0d9b38132caf)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
![Screenshot 2024-10-16 103345](https://github.com/user-attachments/assets/04e5c7b2-7fa3-48a9-bac2-adf558d11ff4)

```
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(xtrain,ytrain)
y_pred=dt.predict(xtest)
print(y_pred)
```
![Screenshot 2024-10-16 103350](https://github.com/user-attachments/assets/1a22b001-9296-4f55-aa5d-eb0959e1a6ad)

```
from sklearn import metrics
mse=metrics.mean_squared_error(ytest,y_pred)
mse
```
![Screenshot 2024-10-16 103355](https://github.com/user-attachments/assets/6c56af55-755a-4b7a-9895-2fc2c534084f)


```
r2=metrics.r2_score(ytest,y_pred)
r2
```
![Screenshot 2024-10-16 103400](https://github.com/user-attachments/assets/78ee3590-302b-421b-8476-ef592d645963)

```
dt.predict([[5,6]])
```
## Output:
![Screenshot 2024-10-16 103407](https://github.com/user-attachments/assets/014964a3-e928-43d6-b0a3-71e18cff477c)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
