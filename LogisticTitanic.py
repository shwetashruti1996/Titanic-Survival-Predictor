import math
import numpy as np 
import pandas as pd 
import seaborn as ans
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def TitanicLogistic():
	#step 1:Load data
	titanic_data = pd.read_csv('TitanicDataset.csv')

	print("First 5 entries from loaded dataset: ")
	print(titanic_data.head())

	print("Number of passengers are : "+str(len(titanic_data)))

	#Step 2:Analyse data
	#Graph on non-servived
	print("Visualisation : Survived and non Survived passengers")
	figure()
	target = "Survived"

	countplot(data=titanic_data,x=target).set_title("Survived and non-Survived passengers")
	show()

	#Graph on gender
	print("Visualisation : Survived and non Survived passengers based on Gender")
	figure()
	target = "Survived"

	countplot(data=titanic_data,x=target, hue="Sex").set_title("Survived and non-Survived passengers based on Gender")
	show()

	#graph on pclass
	print("Visualisation : Survived and non Survived passengers based on passenger class")
	figure()
	target = "Survived"

	countplot(data=titanic_data,x=target, hue="Pclass").set_title("Survived and non-Survived passengers based on passenger class")
	show()

	#graph on Age
	print("Visualisation : Survived and non Survived passengers based on passenger Age")
	figure()
	target = "Survived"

	titanic_data["Age"].plot.hist().set_title("Survived and non-Survived passengers based on Age")
	show()

	#graph on Fare
	print("Visualisation : Survived and non Survived passengers based on passenger Fare")
	figure()
	target = "Survived"

	titanic_data["Fare"].plot.hist().set_title("Survived and non-Survived passengers based on Fare")
	show()

	#Step 3 : Cleaning data
	titanic_data.drop("zero",axis=1,inplace=True)

	print("First 5 records of dataset : ")
	print(titanic_data.head(5))
	
	print("Values of sex column")
	print(pd.get_dummies(titanic_data["Sex"]))

	print("sex column after removing first field : ")
	Sex = pd.get_dummies(titanic_data["Sex"],drop_first = True)
	print(titanic_data.head(5))

	print("Pclass column after removing first field : ")
	Pclass = pd.get_dummies(titanic_data["Pclass"],drop_first = True)
	print(titanic_data.head(5))

	print("Values of data after concatenating new columns")
	titanic_data=pd.concat([titanic_data,Sex,Pclass],axis=1)
	print(titanic_data.head(5))

	print("Values of data after removing irrelevent columns")
	titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
	print(titanic_data.head(5))

	x=titanic_data.drop("Survived",axis=1)
	y=titanic_data["Survived"]

	#STEP 4) Data training
	xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5)
	logmodel = LogisticRegression()
	logmodel.fit(xtrain,ytrain)
	
	#step4) Data testing
	prediction = logmodel.predict(xtest)

	#step5) Calculate Accuracy
	print("Classification report of Logistic Regression : ")
	print(classification_report(ytest,prediction))

	print("Confusion matrix of Logistic Regression : ")
	print(confusion_matrix(ytest,prediction))

	print("Accuracy report of Logistic Regression : ")
	print(accuracy_score(ytest,prediction))



def main():
		TitanicLogistic()

if __name__ == '__main__':
		main()