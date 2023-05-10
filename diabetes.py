# -*- coding: utf-8 -*-
# packages
import numpy as np
import pandas as pd
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

#----------------- dataset loading-----------------------

diabetes_dataset = pd.read_csv('C:\\Users\Rakes\OneDrive\Desktop\Group_Project-01\datasets\diabetes.csv')

#print(diabetes_dataset.groupby("Outcome").mean())

X = diabetes_dataset.drop(columns = "Outcome",axis = 1)
Y = diabetes_dataset["Outcome"]



#-------------------Data standardization----------------

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

#print(standardized_data)




#------------- splitting training and testing data -----------------


X = standardized_data
Y = diabetes_dataset["Outcome"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

#---------------Training the model (Support vector machine) -------------------
classifier = svm.SVC(kernel="linear")

classifier.fit(X_train, Y_train)


# ------------------ Model Evaluation -------------------
# ----------------- Accuracy score -------------------

X_train_prediction = classifier.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
#print(training_data_accuracy)

X_test_prediction = classifier.predict(X_test)

test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
#print(test_data_accuracy)


#----------- Making a predictive system -----------------------

input_data = (8,183,64,0,0,23.3,0.672,32)

# change the input_data to numpy array

input_data_as_numpy_array = np.asarray(input_data)
# reshaping the data
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data

std_data = scaler.transform(input_data_reshaped)
#print(std_data)

prediction = classifier.predict(std_data)

print(prediction)


filename = "diabetes_model.sav"

pickle.dump(classifier,open(filename,"wb"))

loaded_model = pickle.load(open("diabetes_model.sav","rb"))

'''
data = (7,114,66,0,0,32.8,0.258,42)
data_as_numpy_array = np.asarray(data)

data_reshaped = data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(data_reshaped)
print(prediction)


for column in X.columns:
  print(column)
 '''
 

