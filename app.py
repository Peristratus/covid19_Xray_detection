import streamlit as st
from multiapp import MultiApp
import home
import webapp
import hedapp

app = MultiApp()

st.markdown("""
# MediStar Artificial Intelligence diagnosis application

This application is built to utilize a large datasets of disease and train the AI to diagnose certain alignment, as this is not a 100 percent accurate diagnosis tool but can be used to predict the combained datasets to give a ratio or 
true/ false indication of a patients future or current state.

""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("AI Diabetes Detection", webapp.app)
app.add_app("AI HD Detection", webapp.app)
# The main app
app.run()
