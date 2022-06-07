"""main.py"""
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px
st.title('Honest - Data 2021-9')

df = pd.read_csv('../data/2021_9.csv')

