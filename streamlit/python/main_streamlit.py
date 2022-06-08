import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.font_manager import FontProperties
import seaborn as sns
import pydeck as pdk
import plotly.express as px 
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import hashlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


conn = sqlite3.connect('database.db')
c = conn.cursor()

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

def create_user():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_user(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def origin():
	if 'counter' not in st.session_state:
		st.session_state['counter'] = 0


def plus_one_clicks():
	st.session_state['counter'] += 1


def return_int(time):
    if time == '9:':
        return int(9)
    elif time == '7:':
        return int(7)
    elif time == '6:':
        return int(6)
    elif time == '8:':
        return int(8)
    elif time == '4:':
        return int(4)
    elif time == '0.':
        return int(0)
    elif time == '3:':
    	return int(3)
    else:
        return int(time)




menu = ["ログイン","サインアップ"]

choice = st.sidebar.selectbox("メニュー",menu)


if choice == "サインアップ":
	st.subheader("新しいアカウントを作成します")
	new_user = st.text_input("ユーザー名を入力してください")
	new_password = st.text_input("パスワードを入力してください",type='password')

	if st.button("サインアップ"):
		create_user()
		add_user(new_user,make_hashes(new_password))
		st.success("アカウントの作成に成功しました")
		st.info("ログイン画面からログインしてください")

elif choice == "ログイン":
	st.info("please_login")

	username = st.sidebar.text_input("ユーザー名を入力してください")
	password = st.sidebar.text_input("パスワードを入力してください",type='password')
	if st.sidebar.checkbox("ログイン"):
		create_user()
		hashed_pswd = make_hashes(password)

		result = login_user(username,check_hashes(password,hashed_pswd))
		if result:

			st.success("{}さんでログインしました".format(username))

		else:
			st.warning("ユーザー名かパスワードが間違っています")

		if result :
			st.write('sample')
			@st.cache(allow_output_mutation=True)
			def read_file(file):
				df=pd.read_csv(file,engine="python")
				df['伝票日付'] = df['伝票日付'].apply(lambda x: x.replace('/', '-'))
				return df
			
			file_1 = st.sidebar.file_uploader('ファイルアップロード1', type='csv')
			file_2 = st.sidebar.file_uploader('ファイルアップロード２', type='csv')

			file_list = ['file_1', 'file_2']
			chose_list = st.selectbox('選択してください', file_list)

			if chose_list == file_list[0]:
				if file_1 != None:
					df1 = read_file(file_1)
					df1 = df1.dropna()
					df1['伝票日付'] = df1['伝票日付'].apply(lambda x: x[:7])
					df1['year'] = df1['伝票日付'].apply(lambda x: int(x[:4]))	
					df1['month'] = df1['伝票日付'].apply(lambda x: int(x.split('-')[1]))
					df1['day'] = df1['伝票日付'].apply(lambda x: int(x.split('-')[2]))



					button = st.checkbox('show')

					if button == True:
					
						custmers = [i for i in df1['得意先'].unique()] 

						
								
						chose = st.selectbox('顧客を選択してください　↓', custmers)	
						df1 = df1[df1['得意先'] == chose]
									
							
						
						df1_sum = df1.groupby('商品').sum()
						df1_sum['商品'] = df1_sum.index
						df1_month = df1.groupby('month').sum()
						df1_month['month'] = df1_month.index
						b = st.checkbox('data')
						if b == True:
							year = df1['year'].unique()[0]
							st.subheader(f'{chose}: {year}年:商品構成比 (kg)')	
							fig = px.pie(df1_sum, values='正味重量_明細', names='商品')
							st.plotly_chart(fig)
							bottun2 = st.checkbox(f'{year}年月別数量比較')
							if bottun2 == True:
								fig, ax = plt.subplots(1, 1, figsize=(10, 8)) 
								sns.barplot(x='month', y='正味重量_明細',data=df1_month, ax=ax)
								ax.set_ylabel('kg')
								st.pyplot(fig)

			else:
				if file_2 != None:
					df1 = read_file(file_2)
					df1['year'] = df1['伝票日付'].apply(lambda x: int(x[:4]))	
					df1['month'] = df1['伝票日付'].apply(lambda x: int(x.split('-')[1]))
					df1['day'] = df1['伝票日付'].apply(lambda x: int(x.split('-')[2]))



					button = st.checkbox('show')

					if button == True:
					
						custmers = [i for i in df1['得意先'].unique()] 

						
								
						chose = st.selectbox('顧客を選択してください　↓', custmers)	
						df1 = df1[df1['得意先'] == chose]
									
							
						
						df1_sum = df1.groupby('商品').sum()
						df1_sum['商品'] = df1_sum.index
						df1_month = df1.groupby('month').sum()
						df1_month['month'] = df1_month.index
						b = st.checkbox('data')
						if b == True:
							year = df1['year'].unique()[0]
							st.subheader(f'{chose}: {year}年:商品構成比 (kg)')	
							fig = px.pie(df1_sum, values='正味重量_明細', names='商品')
							st.plotly_chart(fig)
							bottun2 = st.checkbox(f'{year}年月別数量比較')
							if bottun2 == True:
								fig, ax = plt.subplots(1, 1, figsize=(10, 8)) 
								sns.barplot(x='month', y='正味重量_明細',data=df1_month, ax=ax)
								ax.set_ylabel('kg')
								st.pyplot(fig)





			
		





