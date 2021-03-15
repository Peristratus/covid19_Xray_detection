import streamlit as st

def app():
    st.title('Home')

    st.write(""" MediStar-Artificial Inteligence Diagnosis software. 
      Diabetes ratio 1=Positive / 0= Negative. 
      Heart Disease number >40 positive with 70 percent accuracy 
      < 40 negative with 70 percent accuracy. 
      Breast Cancer number >13 positive with 90 percent accuracy 
      < 13 negative with 90 percent accuracy. """)
