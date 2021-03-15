import streamlit as st
from multiapp import MultiApp
import home
import webapp
import hedapp
import brcapp

app = MultiApp()

st.markdown("""
# MediStar Artificial Intelligence Diagnosis Application

This application is built as an Artificial Intelligence (AI) diagnostic center, this software is  dependent on Datasets, it can be used to predict the combained datasets per patient to give a ratio or 
true/ false indication of a patients future or current Healt status.

""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("AI Diabetes Detection", webapp.app)
app.add_app("AI Heart Disease Detection", hedapp.app)
app.add_app("AI breast cancer Detection", brcapp.app)
# The main app
app.run()
