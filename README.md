# SGD-Classifier

## Name: Naveen Kanthan L
## Reg no: 21222323018

## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data.
2. Split Dataset into Training and Testing Sets.
3. Train the Model Using Stochastic Gradient Descent (SGD).
4. Make Predictions and Evaluate Accuracy.
5. Generate Confusion Matrix.

## Program:
```python
/*
Program to implement the prediction of iris species using SGD Classifier.
*/
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Load the dataset
iris = load_iris()
```

```python
# Create pandas dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())
```
![image](https://github.com/user-attachments/assets/7ba583f9-4357-42f1-aa90-2ac607c4aed6)

```python
x = df.drop('target', axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(x_train, y_train)
```
![image](https://github.com/user-attachments/assets/72368439-81fd-4d4a-949b-1a7c732d79d5)

```python
cf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cf)
```
## Output:
![image](https://github.com/user-attachments/assets/f780a9cd-8433-4196-8dc4-6ec4f47eb85b)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
