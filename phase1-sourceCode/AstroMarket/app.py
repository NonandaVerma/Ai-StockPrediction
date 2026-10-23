"""Market Prediction dashboard.  Run:  streamlit run app.py"""
import streamlit as st

from src.config import load_config
from src.ui_forecast import render

st.set_page_config(page_title="Market Prediction", page_icon="📈", layout="wide")
render(load_config())
