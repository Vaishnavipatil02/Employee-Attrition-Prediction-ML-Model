import pandas as pd
import numpy as np
import matplotlib.pyplot  
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv ("Employee-Attrition.csv")

print(df.head())

df.drop (['EmployeeNumber','Over18','StandardHours','EmployeeCount','MonthlyRate','PercentSalaryHike'],axis=1,inplace=True)

for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column] = LabelEncoder().fit_transform(df[column])

X = df.drop(['Attrition'], axis = 1)
Y = df['Attrition']

X_train, X_test, Y_train , Y_test = train_test_split(X, Y, test_size= 0.20, random_state=0)

#feature scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

random_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

random_forest.fit(X_train, Y_train)

# Saving model to disk
pickle.dump(random_forest, open('model.pkl','wb'))
