#Iris - Data Set
import pandas #data frame
from sklearn.tree import DecisionTreeClassifier #Self Explanatory
from sklearn.model_selection import train_test_split #Split data set to test and train
from sklearn.metrics import accuracy_score #For calculating accuracy

iris_data=pandas.read_csv('Iris.csv')
x=iris_data.drop(columns=["Id","Species"])
y=iris_data["Species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
score=accuracy_score(y_test,predictions)
print("Model built successfully with an accuracy of ",(score*100),"%")