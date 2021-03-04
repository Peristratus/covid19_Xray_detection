import glob
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matpolotlib.pyplot as plt
import keras as k



#create a title and sub-title
st.write("""

# Kidney Disease Detection
Detect if someone has Kidney Disease using Artificial Intelligence 

""")
#open and display Image

image =Image.open('/workspace/covid19_Xray_detection/dataset/kidney_disease.csv')
st.image(image, caption='ML' , use_column_width=True)

#get the shape of the data (the number of rows and cols)
df.shape

#create a list of colum names to keep
columns_to_retain = ['sg', 'al', 'sc', 'hemo', 'pcv', 'wbbc', 'rbcc', 'htn', 'classification']

#drop the columns that are not in columns_to_retain
df = df.drop([col for col in df.columns if not col in columns_to_retain], axis= 1 )

#drop the rows with na or missing values
df = df.dropna(axis=0)