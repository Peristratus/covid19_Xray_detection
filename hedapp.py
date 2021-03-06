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

image =Image.open('/workspace/covid19_Xray_detection/dataset/aiheart.jpg')
st.image(image, caption='ML' , use_column_width=True)

#Get the data
df = pd.read_csv('/workspace/covid19_Xray_detection/dataset/heart_statlog_cleveland_hungary_final.csv')
#set a subheader
st.subheader('Data Information')
#show the data as a table
st.dataframe(df)
#show statistics on the data
st.write(df.describe())
#show the data as a chart
chart = st.bar_chart(df)

#split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:11].values
Y = df.iloc[:, 0:1].values.ravel()
#split the data set into 75% Traning and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#get the feature impot from the user 
def get_user_input():
    age = st.sidebar.slider('age', 16, 90, 25)
    sex = st.sidebar.slider('sex', 0, 1, 0)
    chest_pain = st.sidebar.slider('chest_pain', 1, 4, 0)
    resting_bp = st.sidebar.slider('resting_bp', 0, 200, 0)
    cholesterol = st.sidebar.slider('cholesterol', 0,700,0 )
    fasting_blood_sugar = st.sidebar.slider('fasting_blood_sugar', 0, 1, 0)
    resting_ecg = st.sidebar.slider('resting_ecg', 0, 2, 0)
    max_heart_rate = st.sidebar.slider('max_heart_rate', 71, 202, 73)
    exercise_angina = st.sidebar.slider('exercise_angina', 0, 1, 0)
    ST_slope = st.sidebar.slider('ST_slope', 0, 2, 0)
    oldpeak = st.sidebar.slider('oldpeak', 0.0, 3.0, 1.5)
     

#store a dictionary into a variable
    user_data ={'age': age,
               'sex':sex,
                'chest_pain':  chest_pain,
                'resting_bp': resting_bp,
                 'cholesterol': cholesterol,
                 'fasting_blood_sugar': fasting_blood_sugar,
                 'resting_ecg':  resting_ecg,
                 'max_heart_rate': max_heart_rate,
                 'exercise_angina': exercise_angina,
                 'ST_slope': ST_slope,
                 'oldpeak': oldpeak,

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
