import streamlit as st
from multiapp import MultiApp
import home
import webapp

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("webapp", webapp.app)

# The main app
app.run()