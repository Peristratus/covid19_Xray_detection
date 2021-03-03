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
X = 
