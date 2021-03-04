import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st



#create a title and sub-title
st.write("""

# Heart Disease
Detect if someone has Heart Disease using Artificial Intelligence 

""")
#open and display Image

image =Image.open('/workspace/covid19_Xray_detection/dataset/ai.jpg')
st.image(image, caption='ML' , use_column_width=True)

#Get the data
df = pd.read_csv('/workspace/covid19_Xray_detection/dataset/heart_failure_clinical_records_dataset.csv')
#set a subheader
st.subheader('Data Information')
#show the data as a table
st.dataframe(df)
#show statistics on the data
st.write(df.describe())
#show the data as a chart
chart = st.bar_chart(df)

#split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:13].values
Y = df.iloc[:, -1].values
#split the data set into 75% Traning and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#get the feature impot from the user 
def get_user_input():
    anaemia = st.sidebar.slider('anaemia', 0, 1, 0)
    creatinine_phosphokinase = st.sidebar.slider('creatinine_phosphokinase', 20, 3000, 199)
    ejection_fraction = st.sidebar.slider('ejection_fraction', 0,100,50 )
    high_blood_pressure = st.sidebar.slider('high_blood_pressure', 0, 0, 1)
    platelets = st.sidebar.slider('platelets', 30000, 800000, 5000)
    serum_creatinine = st.sidebar.slider('serum_creatinine', 0.1, 9.0, 0.15)
    serum_sodium = st.sidebar.slider('serum_sodium', 50, 150, 90)
    diabetes = st.sidebar.slider('diabetes', 0, 1, 0)
    sex = st.sidebar.slider('sex', 0, 1, 0)
    smoking = st.sidebar.slider('smoking', 0, 1, 0)
    time = st.sidebar.slider('time', 0, 289, 100)
    age = st.sidebar.slider('age', 16, 90, 25)
    DEATH_EVENT = st.sidebar.slider('DEATH_EVENT', 0, 1, 0)

#store a dictionary into a variable
    user_data ={'anaemia': anaemia,
               'creatinine_phosphokinase':creatinine_phosphokinase,
                ' ejection_fraction':  ejection_fraction,
                ' high_blood_pressure': high_blood_pressure,
                 ' platelets': platelets,
                 ' serum_creatinine': serum_creatinine,
                 ' serum_sodium': serum_sodium,
                 ' sex': sex,
                 'smoking':smoking,
                 'time': time,
                 'diabetes': diabetes,
                 'age': age,
                 'DEATH_EVENT':DEATH_EVENT,

            }
    #Transform the data into data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

#store the user the users input into a variable
user_input = get_user_input()

#set a sub header and display the users input
st.subheader('User Input:')
st.write(user_input)

#create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,  Y_train )

#show the model matrix
st.subheader('Model Test Accuracy Score:')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

#store model prediction ina variable
prediction = RandomForestClassifier.predict(user_input)

#set a subheader display the classification
st.subheader('clasification:')
st.write(prediction)