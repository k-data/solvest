""" data_get.py """
import pandas as pd
import streamlit as st

class Data_Get():
    def read_data(self):
        train_raw = pd.read_csv('../data/2021b.csv') 
        print('The size of the 2021b data:' + str(train_raw.shape))

        print('Target column = "Bブロック"')
        print('Target mean:' + str(train_raw['Bブロック'].mean()))

        df = train_raw.copy()
       
        print(df.describe(include='all'))
        return train_raw
        