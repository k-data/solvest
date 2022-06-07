""" data_get.py """
import pandas as pd
import streamlit as st

class Data_Get():
    def read_data(self):
        train_raw = pd.read_csv('../data/2021b.csv') 
        st.write('The size of the 2021b data:' + str(train_raw.shape))
        
        target = train_raw.columns
        sel_target = st.sidebar.selectbox('(1) Please select "Target column"',target)

        # st.write('Target mean:' + str(round(train_raw[sel_target].mean(), 3)))

        # df = train_raw.copy()

        # checkbox = st.checkbox('Show data')
        # if checkbox:
        #     checkbox_1 = st.checkbox('2021_data')
        #     if checkbox_1:
        #         st.write(df.describe(include='all'))
       
        
        # return train_raw
        