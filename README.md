# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Ashwin Kumar A
RegisterNumber: 212223040021 
```
```py
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![image](https://github.com/user-attachments/assets/58be1da8-ab91-4ebf-8976-72cae1c192fa)

![image](https://github.com/user-attachments/assets/1f655509-e4f8-4a00-a93d-7b3002fd3c03)

![image](https://github.com/user-attachments/assets/686fc30f-b9f5-40a3-972e-9030cdb7064d)

![image](https://github.com/user-attachments/assets/0a76cdb8-9ff6-4a94-9d00-4413c4cd2a1e)

![image](https://github.com/user-attachments/assets/bfdc4e99-9d04-4585-a7c9-e4b77a5798a8)

![image](https://github.com/user-attachments/assets/467e6ab1-f6a6-48e4-ac06-a2bcb6ca8de1)

![image](https://github.com/user-attachments/assets/c813b96a-5977-4307-8c77-8a24354b3395)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
