""" main.py """

import pandas as pd
import streamlit as st
import data_get as dg

st.title('Honest - Data from 2021')

st.subheader('(1) Data Structure')
dg_Inst = dg.Data_Get()
train_raw = dg_Inst.read_data()