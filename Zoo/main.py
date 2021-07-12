#### Training data provider https://archive.ics.uci.edu/ml/datasets/Zoo

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import random 
import json

#################################################################
def count_type(data, column):
    sum = 0
    for i in range(0,100):
        if data._get_value(i, column) == 1:
            sum += 1
    #print("Sum of " + column + " animals is: " + str(sum))
    return sum



def get_some_random_animal_stats(list_of_columns, my_data):
    number_of_rolls = random.randint(4, 10)
    my_list = [[],[]]
    tmp_list = list_of_columns[1:13] + list_of_columns[14:17]
    random.shuffle(tmp_list)
    my_list[0] = tmp_list[0:number_of_rolls]
    for i in range(0, len(my_list[0])):
        my_list[1].append( str( count_type(my_data, my_list[0][i]) ) )
    return my_list    


def get_output_names(output_number, list_of_names):
    my_num = output_number - 1
    return list_of_names[my_num]


def get_data_from_user(list_of_column_names):
    list_of_labels = list_of_column_names[1:17]
    list_of_data = []
    my_str_data = "{ "
    for i in range(0,12):
        my_str_data += ' "{}" : "{}", '.format(list_of_labels[i], int(st.sidebar.checkbox(list_of_labels[i])))
    my_str_data += ' "{}" : "{}", '.format(  list_of_labels[12], int(  st.sidebar.slider(list_of_labels[12], 0, 8, 2)  )  )
    my_str_data += ' "{}" : "{}", '.format(list_of_labels[13], int(st.sidebar.checkbox(list_of_labels[13])))
    my_str_data += ' "{}" : "{}", '.format(list_of_labels[14], int(st.sidebar.checkbox(list_of_labels[14])))
    my_str_data += ' "{}" : "{}" '.format(list_of_labels[15], int(st.sidebar.checkbox(list_of_labels[15])))
    my_str_data += " }" 
    jsonForm = json.loads(my_str_data)
    #mypd = pd.read_json(jsonForm)
    mypd = pd.json_normalize(jsonForm)
    features = pd.DataFrame(mypd, index = [0])
    return features
    #print(jsonFromList)
##################################################


data = pd.read_csv("zoo.csv", sep = ";")
output_group = ["mammals", "birds", "reptiles", "fish", "amphibians", "bugs", "arthropods/mollusca"]
list_of_column_names = list(data.columns)





st.write("""
    #Animal zoo clasifier -
    Python machine learning animal casifier
""")

st.subheader("ANIMALS")
st.dataframe(data)

st.subheader("some animal statistics:")
data_stats = get_some_random_animal_stats(list_of_column_names, data)

for i in range(0, len(data_stats[0])):
    st.write( data_stats[0][i] + ": " + data_stats[1][i] )



X = data.iloc[:, 1:17].values
Y = data.iloc[:, -1].values
Y2 = data.iloc[:,0].values




#tests - 0.25, train size - 0.75
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y2, test_size = 0.25, random_state = 0)

#data input


user_input = get_data_from_user(list_of_column_names)
st.subheader("User input: ")
st.write(user_input)

#training model - guessing type of animal

randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(X_train, Y_train)
#training model - guessing species of animal

randomForestClassifier2 = RandomForestClassifier()
randomForestClassifier2.fit(X_train2, Y_train2)
#########
st.subheader("Accuracy of trained model 1")
st.write( str( accuracy_score(Y_test, randomForestClassifier.predict(X_test)) ) )
prediction = randomForestClassifier.predict(user_input)
prediction2 = randomForestClassifier2.predict(user_input)
st.subheader("Classifier prediction - type: ")
st.write(str( get_output_names(prediction.item(0), output_group) ))

###########
st.subheader("Accuracy of trained model 2")
st.write( str( accuracy_score(Y_test2, randomForestClassifier2.predict(X_test)) ) )
st.subheader("Classifier prediction - species: ")
st.write(prediction2)
#st.write( str(prediction.item(0)) )

