import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
from multiapp import MultiApp


def app():
    global RandomForestClassifier
    st.title(" AI breast cancer Detection")
    st.write("""
    
    Detect if someone has breast cancer using Artificial Intelligence 

    """)

    image =Image.open('dataset/AI_beca.jpg')
    st.image(image, caption='ML' , use_column_width=True)

    #Get the data
    df = pd.read_csv('dataset/data1.csv')
    #set a subheader
    st.subheader('Data Information')
    #show the data as a table
    st.dataframe(df)
    #show statistics on the data
    st.write(df.describe())
    #show the data as a chart
    chart = st.bar_chart(df)

    #split the data into independent 'X' and dependent 'Y' variables
    X = df.iloc[:, 0:14].values
    Y = df.iloc[:, 0:1].values.ravel()
    Y=Y.astype('int')
    #split the data set into 75% Traning and 25% testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    #get the feature impot from the user 
    def get_user_input():
        radius_mean = st.sidebar.slider('radius_mean', 0.0, 20.0, 9.5)
        texture_mean = st.sidebar.slider('texture_mean', 0.0, 30.0, 11.0)
        perimeter_mean = st.sidebar.slider('perimeter_mean', 0, 200, 120)
        area_mean = st.sidebar.slider('area_mean', 0, 3000, 230) 
        smoothness_mean = st.sidebar.slider('smoothness_mean', 0.0, 0.20, 0.13)
        compactness_mean = st.sidebar.slider('compactness_mean', 0.0, 0.5, 0.21)
        concavity_mean = st.sidebar.slider('concavity_mean', 0.0, 0.5 , 0.10)
        concave_points_mean = st.sidebar.slider('concave_points_mean', 0.0, 0.5 , 0.10)
        symmetry_mean = st.sidebar.slider('symmetry_mean', 0.1, 0.5, 0.1)
        fractal_dimension_mean= st.sidebar.slider('fractal_dimension_mean', 0.0, 0.5, 0.1)
        radius_se= st.sidebar.slider('radius_se', 0.1, 2.0, 0.1)
        texture_se= st.sidebar.slider('texture_se', 0.1, 2.0, 0.1)
        perimeter_se= st.sidebar.slider('perimeter_se', 1.0, 10.0, 1.5)
        area_se = st.sidebar.slider('area_se', 0, 220, 11)

    #store a dictionary into a variable
        user_data ={'radius_mean': radius_mean,
                 'texture_mean':texture_mean,
                 ' perimeter_mean': perimeter_mean,
                 'area_mean':  area_mean,
                 'smoothness_mean': smoothness_mean,
                 'compactness_mean':compactness_mean,
                 'concavity_mean':concavity_mean,
                 'concave_points_mean':concave_points_mean,
                 'symmetry_mean':symmetry_mean,
                 'fractal_dimension_mean':fractal_dimension_mean,
                 'radius_se': radius_se,
                 'texture_se':texture_se,
                 'perimeter_se':perimeter_se,
                 'area_se':area_se
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
    Random = RandomForestClassifier()
    Random.fit(X_train,Y_train )

    #show the model matrix
    st.subheader('Model Test Accuracy Score:')
    st.write( str(accuracy_score(Y_test, Random.predict(X_test)) * 100) + '%')

    #store model prediction ina variable
    prediction = Random.predict(user_input)

    #set a subheader display the classification
    st.subheader('clasification:')
    st.write(prediction)

