import pandas as pd
data = pd.read_csv("dataset/pd_speech_features.csv")
print('Columns in Data', data.columns)
#check the shape of the data
print("Parkinsons data has %d rows and %d columns" % data.shape)
print(data.head(5))
