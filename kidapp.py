import glob
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import keras as k
import streamlit as st



#create a title and sub-title
st.write("""

# Kidney Disease Detection with data sets
Detect if someone has diabetes using Artificial Intelligence 

""")

#get the shape of the data (the number of rows and cols)
df = pd.read_csv('/workspace/covid19_Xray_detection/dataset/kidney_disease.csv')

#set a subheader
st.subheader('Data Information')
#show the data as a table
st.dataframe(df)
#show statistics on the data
st.write(df.describe())
#show the data as a chart
chart = st.bar_chart(df)

#create a list of colum names to keep
columns_to_retain = ['sg', 'al', 'sc', 'hemo', 'pcv', 'wbbc', 'rbcc', 'htn', 'classification']

#drop the columns that are not in columns_to_retain
df = df.drop([col for col in df.columns if not col in columns_to_retain], axis= 1 )

#drop the rows with na or missing values
df = df.dropna(axis=0)

#transform the non numeric data in the columns
for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column] = LabelEncoder().fit_transform(df[column])

#split the data into independent (X) dataset (the feature) and dependent (Y) data set

X = df.drop(['classification'], axis=1)
y = df['classification']

#feature scaling
#min-max scaler method scales that data set so that all the input features lie between 0 and 1
x_scaler = MinMaxScaler()
x_scaler.fit(X)
column_names = X.columns
X[column_names] = x_scaler.transform(X)

#Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)

#Build the model
model = Sequential()
model.add(Dense(256, input_dim= len(X.columns) , kernel_initializer= k.initializers.random_normal(seed=13), activation='relu'))
model.add(Dense(1, activation='hard_sigmoid'))

#compile the model
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

#train the model
history= model.fit(X_train, y_train, epochs = 2000, batch_size = X_train.shape[0])

#save the model
model.save('ckd.model')

# visuals model loss and accuracy
plt.plot(history.history['loss'])
plt.title('model accuracy & loss')
plt.ylabel('accuracy and loss')
plt.xlabel('epoch')

print('shape of training data:', X_train.shape)
print('shape of test data:', X_test.shape)

pred = model.predict(X_test)
pred = [1 if y>=0.5 else 0 for y in pred]

print('original : {0}'.format(",".join(str(x) for x in y_test )))
print('Predicted : {0}'.format(",".join(str(x) for x in pred )))

#set a subheader display the classification
st.subheader('clasification:')
st.write(y_test)
st.write(pred)
st.write(df.dropna(axis=0))



