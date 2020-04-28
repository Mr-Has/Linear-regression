import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Reading the data
data = pd.read_csv("student-mat.csv", sep = ";")
# Print the five first element
print(data.head())

# Trimming the data to what we want (only the attribute that we want)
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# To check the data after trimming
print("\n",data.head())

# Now we define what values we want to predict (This is also called label)
predict = "G3"

# Setting two arrays, one to defined our attributes and other to define labels
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#Splitting these into 4 variables(x test, y test, x train, y train)
# Here we split 10% of the data for training
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Creating linear model and check how well it works
linear = linear_model.LinearRegression()
# It fits the data to find the best line (best line on graph among the data)
linear.fit(x_train, y_train)
# To show the accuracy of the model
acc = linear.score(x_test, y_test)
print("\n", acc)

# Until here we created the model
# How to use the model

# To print the list of coefficients of the five variables
print("\nCoefficient: \n" , linear.coef_)
print ("\nIntercept: \n" , linear.intercept_)

# How to predict the student performance (the data is about students)
prediction = linear.predict(x_test)
for x in range(len(prediction)):
# Printing the prediction, input data(corelated to x_test[x]), actual value of final grade
    print("\n",prediction[x], x_test[x], y_test[x])

#(https://www.youtube.com/watch?v=1BYu65vLKdA)