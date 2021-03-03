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

image =Image.open('/workspace/covid19_Xray_detection/dataset/ai.png')
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
