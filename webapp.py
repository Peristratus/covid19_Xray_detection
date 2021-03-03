import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st



#create a title and sub-title
st.write("""

# Diabetes Detection
Detect is someone has diabetes using machine learning

""")
#open and display Image

image =Image.open('/workspace/covid19_Xray_detection/dataset/ai.jpg')
st.image(image, caption='ML' , use_column_width=True)

#Get the data
df = pd.read_csv('/workspace/covid19_Xray_detection/dataset/diabetes.csv')
#set a subheader
st.subheader('Data Information')
#show the data as a table
st.dataframe(df)
#show statistics on the data
st.right(df.describe())
#show the data as a chart
chart = st.bar_chart(df)

#split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values
#split the data set into 75% Traning and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#get the feature impot from the user 
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

#store a dictionary into a variable
    user_data ={'pregnancies': pregnancies,
               'glucose':glucose,
               'blood_presure':blood_pressure,
                ' skin_thickness':  skin_thickness,
                ' insulin':  insulin,
                'BMI':BMI,
                'DPF':DPF,
                'age':age,
            }
    #Transform the data into data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

#store the user the users input into a variable
user_input = get_user_input()

#set a sub header and display the users input
st.subheader('User Input:')
st.write(user_input)


#store the user input into the variable
user_input = get_user_input()


#set a subheader and display the users input
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