# Kaggle Competetion - Predicion of survivors in Titanic crash

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing training dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#Encoding the name feature
train_test_data = [train, test] 

for dataset in train_test_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')
for dataset in train_test_data:
   dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

   dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
   dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
   dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_map = {"Master":1,"Miss":2,"Mr":3,"Mrs":4,"Other":5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_map)
    
#Encoding sex feature {Integer Encoding} (Test-1)
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map({"male":1,"female":0})    

#Taking care of missing data in Age
for dataset in train_test_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    #Filling Gaps with random nos bw sum and diff of mean and sd
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4    
    
#Encoding Embarked {Test-1}    
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map({"S":0,"Q":2,"C":1})
    
#Filling Embarked  Values
for dataset in train_test_data:
    emb_null_count = train['Embarked'].isnull().sum()
    emb_null_random_list = np.random.randint(0,2, size=emb_null_count)
    dataset['Embarked'][np.isnan(dataset['Embarked'])] = emb_null_random_list
   
#Combining Siblings and Parents to Familysize
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']

features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
for dataset in train_test_data:
    fare_avg = dataset['Fare'].mean()
    fare_std = dataset['Fare'].std()
    fare_null_count = dataset['Age'].isnull().sum()
    #Filling Gaps with random nos bw sum and diff of mean and sd
    dataset['Fare'][np.isnan(dataset['Fare'])] = fare_avg + fare_std
    dataset['Fare'] = dataset['Fare'].astype(int)
    
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)    

train = train.drop(features_drop,axis=1) 
test = test.drop(features_drop,axis=1)
train = train.drop('PassengerId',axis=1)  

#Creating training and test set
X_train = train.drop('Survived',axis=1)
y_train = train['Survived']
X_test = test.drop('PassengerId',axis=1)

#Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''
 
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
acc_log_reg = round( classifier.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent')

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })

submission.to_csv('submission.csv', index=False)



    
    
    
    
    
    
    
    
    

                       
