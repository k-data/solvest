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

def create_data():
    create_data = """
                CREATE TABLE IF NOT EXISTS file1
                (id INTEGER, day TEXT, custmers TEXT, product TEXT, kg INTEGER, price INTEGER, revenue INTEGER)
                  """
    c.execute(create_data)


def add_data():
    insert_data = """
                  INSERT INTO file1 VALUES(?,?,?,?,?,?,?)
                  """

    for idx, row in df1.iterrows():
        c.execute(insert_data,(row))
    c.commit()


value_kongou = ['混合廃棄物（フライト）','混合廃棄物（がれき類）','混合廃棄物Ａ　','混合廃棄物Ａ（Retail）','混合廃棄物Ａ（Marｌceting）','混合廃棄物Ａ（MV）','混合廃棄物Ａ（IT）','混合廃棄物Ａ（EC）', '混合廃棄物Ａ（CS）','混合廃棄物（金属くず・廃プラ・ガラ陶）','混合廃棄物（金属くず・廃プラ）','混合廃棄物（木くず・廃プラ）','ｶﾞﾗｽ・ｺﾝｸﾘｰﾄ・陶磁器くず','混合廃棄物\u3000（フライト）','混合廃棄物（木くず・廃プラ）','混合廃棄物（ボード混入）','混合廃棄物（処理困難物）', '混合廃棄物','混合廃棄物Ａ', '混合廃棄物Ｂ', '混合廃棄物Ｃ', '混合廃棄物（安定型）', '混合廃棄物 （ビン・缶・ペットボトル）', '混合廃棄物（ビン・缶・ペットボトル）']
value_shoukyaku = ['混合廃棄物（焼却物）.','混合廃棄物（焼却物）  ','ビデオテープ','混合廃棄物（布団）','混合廃棄物（靴）','混合廃棄物（反物）','混合廃棄物（臭気物）','混合廃棄物（ソファー）', '混合廃棄物（焼却物）','混合廃棄物（壁紙）']
value_keiryou = ['軽量物系　Ｂ（グラスウール）  ','軽量物系　Ａ（スタイロフォーム） ','発泡スチロール','軽量物系　Ｂ（スタイロフォーム）','軽量物系\u3000Ａ（岩綿吸音板）','軽量物系\u3000Ａ（スタイロフォーム）  ','軽量物系\u3000Ｂ（充填材）','軽量物系\u3000Ｂ（ＦＲＰ）','軽量物系\u3000Ａ（発泡スチロール）', '軽量物系\u3000Ｂ（グラスウール）', '軽量物系\u3000Ａ（ネオマフォーム）', '軽量物系\u3000Ａ（スタイロフォーム）','軽量物系\u3000Ａ（ウレタン）', '軽量物系\u3000Ａ（スポンジ）']
value_kikuzu = ['伐根材','木くず','木くずA','生木']
value_borad = ['石膏ボード　Ｃ ','石膏ボード　Ｄ  ','石膏ボード　Ａ ','パ－テ－ションボ－ド','石膏ボード\u3000Ｃ','石膏ボード\u3000Ａ','石膏ボード\u3000Ｄ','石膏ボード\u3000Ｂ']
value_haipura = ['廃プラスチック','廃プラスチック類（箱入りシール）','廃プラスチック類','廃プラスチック類','廃プラスチック類（箱入りシール）']
value_kinzoku = ['選別  ','鉄千地','ＧＤ\u3000','ＧＣ\u3000軽鉄・スチ－ル類', '室外機','室内機','選別','金庫','金属くず','選別\u3000','選別（キャビネット）']



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
			st.write('データをアップロードしてください')
			@st.cache(allow_output_mutation=True)
			def read_file(file):
				df=pd.read_csv(file,engine="python")
				df['伝票日付'] = df['伝票日付'].apply(lambda x: x[:-4])
				df['伝票日付'] = df['伝票日付'].apply(lambda x: x.replace('/', '-'))
				
				return df
			
			file_1 = st.sidebar.file_uploader('ファイルアップロード1', type='csv')
			file_2 = st.sidebar.file_uploader('ファイルアップロード2', type='csv')
			file_3 = st.sidebar.file_uploader('ファイルアップロード3', type='csv')
			file_list = ['file_1', 'file_2','file_3']
			chose_list = st.sidebar.select_slider('ファイルを選択してください', file_list)

			if chose_list == file_list[0]:
				if file_1 != None:
					df1 = read_file(file_1)
					
					df1 = df1[['伝票番号','伝票日付','得意先','商品','集計区分','正味重量_明細','金額']]
					df1 = df1.dropna()
					
					create_data()
					add_data()

					custmers = [i for i in df1['得意先'].unique()] 
					df1['year'] = df1['伝票日付'].apply(lambda x: int(x[:4]))	
					df1['month'] = df1['伝票日付'].apply(lambda x: int(x.split('-')[1]))
					df1['day'] = df1['伝票日付'].apply(lambda x: int(x.split('-')[2]))


					button = st.checkbox('show')

					if button == True:
						year = df1['year'].unique()
						month = df1['month'].unique()
						kg_all = df1['正味重量_明細'].sum()
						kg_mean = kg_all / len(df1['day'].unique())
						st.write(f'{year}年 {month}月')
						st.write(f'顧客数:{len(custmers)} 搬入量:{kg_all}kg  日量平均:{kg_mean:.2f}kg') 
						df1_bar = df1.groupby('day').sum()
						st.bar_chart(df1_bar[['正味重量_明細','金額']])
						colum_list = ['正味重量_明細', '金額']
						sel_col = st.select_slider('選択してください', colum_list)
						df1_all = df1.groupby('得意先').sum()
						df1_all['得意先'] = df1_all.index
						df1_all = df1_all.iloc[:,:2]
						st.write(f'{sel_col}上位順')
						df1_all = df1_all.sort_values(sel_col, ascending=False)
						st.area_chart(df1_all[sel_col].values)
						with st.container():
							col1, col2 = st.columns([1, 1])
						with col1:
							st.write(df1_all)
						with col2:
							df1_v = df1.groupby('集計区分').sum()
							st.bar_chart(df1_v[['正味重量_明細']])
						value_list = []
						
						for i in df1['商品'].values:
							if i in value_kongou:
								value_list.append('混合系')
							elif i in value_keiryou:
								value_list.append('軽量系')
							elif i in value_shoukyaku:
								value_list.append('焼却系')
							elif i in value_kikuzu:
								value_list.append('木くず')
							elif i in value_kinzoku:
								value_list.append('金属系')
							elif i in value_borad:
								value_list.append('石膏ボード系')
							elif i in value_haipura:
								value_list.append('廃プラスチック系')
							else:
								value_list.append('その他')
						df1['分類'] = value_list

						type_list = [i for i in df1['集計区分'].unique()]
						select_type = st.sidebar.select_slider('種類を選択してください', type_list)
						df1 = df1[df1['集計区分'] == select_type]
						select_df = df1.groupby('得意先').sum()
						select_df = select_df.sort_values(sel_col, ascending=False)
						type_df = df1.groupby('分類').sum()
						type_df['分類'] = type_df.index
						fig = px.pie(type_df, values=sel_col, names='分類',title=f'{select_type}:{sel_col}構成比')
						st.plotly_chart(fig)
						product = st.select_slider('選択してください', type_df.index)
						product_df = df1[df1['分類'] == product]
						product_df = product_df.groupby('商品').sum()
						st.bar_chart(product_df[sel_col])
						st.write(f'{select_type} 上位順')
						st.write(select_df[colum_list])
						custmer = st.text_input('顧客名を入力してください')
						custm_list = []
						for i, v in enumerate(custmers):
							if custmer in v:
								custm_list.append(v)		
						chose = st.selectbox('顧客を選択してください ↓',custm_list)
						df1 = df1[df1['得意先'] == chose]
						month_list = [i for i in df1['month'].unique()]
							
						df1_valance = df1.groupby('集計区分').sum()
						df1_valance['集計区分'] = df1_valance.index
						df1_sum = df1.groupby('商品').sum()
						df1_sum['商品'] = df1_sum.index
						df1_day = df1.groupby('day').sum()
						df1_day['day'] = df1_day.index
						b = st.checkbox('data')
						if b == True:
							if sel_col == colum_list[0]:
								product_list = [i for i in df1_sum['商品'].unique()]
								options = st.sidebar.multiselect('商品を選択してください',product_list, product_list)
								df1_sum = df1_sum.loc[options]
								
								value = '正味重量_明細'
								st.write(df1_sum.iloc[:,:2])
								st.bar_chart(df1_sum.iloc[:,0])
								fig = px.pie(df1_sum, values=value, names='商品')
								fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
								st.plotly_chart(fig)
								data_check = st.checkbox('check')
								if data_check == True:
									if len(month_list) == 1:
										st.write(f'{month_list[0]}月 日量推移')
										st.bar_chart(df1_day[sel_col])
										sum_kg = df1_day['正味重量_明細'].sum()
										st.write(f'{month_list[0]}月 合計{sum_kg}kg')
									else:
										choice_month = st.select_slider('選択してください', month_list, options=month_list[0])
										df1 = df1[df1['month'] == choice_month]
										df1_day = df1.groupby('day').sum()
										st.write(f'{choice_month}月 日量推移')
										st.bar_chart(df1_day[sel_col])
										sum_kg = df1_day[value].sum()
										st.write(f'{choice_month}月 合計{sum_kg}kg')
							
							else:
								value = '金額'
								if len(df1_sum['商品']) < 15:
									st.write(f'{chose}: データ全体 :商品構成比 {colum_list[1]}')	
									fig = px.pie(df1_sum, values=value, names='商品')
									st.plotly_chart(fig)
								else:
									st.bar_chart(df1_sum.loc[:,value])

								data_check = st.checkbox('check')
								if data_check == True:
									if len(month_list) == 1:
										st.write(f'{month_list[0]}月 売上金額推移')
										st.bar_chart(df1_day[sel_col])
										sum_kg = df1_day[value].sum()
										st.write(f'{month_list[0]}月 合計{sum_kg}円')
									else:
										choice_month = st.select_slider('選択してください', month_list, options=month_list[0])
										df1 = df1[df1['month'] == choice_month]
										df1_day = df1.groupby('day').sum()
										st.write(f'{choice_month}月 売上推移')
										st.bar_chart(df1_day[sel_col])
										sum_kg = df1_day[value].sum()
										st.write(f'{choice_month}月 合計{sum_kg}円')



			elif chose_list == file_list[1]:
				if file_2 != None:
					df2 = read_file(file_2)
					df2 = df2[['伝票番号','伝票日付','得意先','商品','集計区分','正味重量_明細','金額']]
					df2 = df2.dropna()
					con = sqlite3.connect('file2.db')
					cursor = con.cursor()
					create_data = """CREATE TABLE IF NOT EXISTS file2
							   (id INTEGER, day TEXT, custmers TEXT, product TEXT, type TEXT, kg INTEGER, revenue INTEGER)
							   """
					cursor.execute(create_data)
					insert_data = """
					              INSERT INTO file1 VALUES(?,?,?,?,?,?,?)
					              """
	
					for idx, row in df2.iterrows():
						cursor.execute(insert_data,(row))
					con.commit()
					
					custmers = [i for i in df2['得意先'].unique()] 

					df2['year'] = df2['伝票日付'].apply(lambda x: int(x[:4]))	
					df2['month'] = df2['伝票日付'].apply(lambda x: int(x.split('-')[1]))
					df2['day'] = df2['伝票日付'].apply(lambda x: int(x.split('-')[2]))


					button = st.checkbox('show')

					if button == True:
						year = df2['year'].unique()
						month = df2['month'].unique()
						kg_all = df2['正味重量_明細'].sum()
						kg_mean = kg_all / len(df2['day'].unique())
						st.write(f'{year}年 {month}月')
						st.write(f'顧客数:{len(custmers)} 搬入量:{kg_all}kg  日量平均:{kg_mean:.2f}kg') 
						df2_bar = df2.groupby('day').sum()
						st.bar_chart(df2_bar[['正味重量_明細','金額']])
						colum_list = ['正味重量_明細', '金額']
						sel_col = st.select_slider('選択してください', colum_list)
						df2_all = df2.groupby('得意先').sum()
						df2_all['得意先'] = df2_all.index
						df2_all = df2_all.iloc[:,:2]
						st.write(f'{sel_col}上位順')
						df2_all = df2_all.sort_values(sel_col, ascending=False)
						st.area_chart(df2_all[sel_col].values)
						with st.container():
							col1, col2 = st.columns([1, 1])
						with col1:
							st.write(df2_all)
						with col2:
							df2_v = df2.groupby('集計区分').sum()
							st.bar_chart(df2_v[['正味重量_明細']])
						value_list = []
						
						for i in df2['商品'].values:
							if i in value_kongou:
								value_list.append('混合系')
							elif i in value_keiryou:
								value_list.append('軽量系')
							elif i in value_shoukyaku:
								value_list.append('焼却系')
							elif i in value_kikuzu:
								value_list.append('木くず')
							elif i in value_kinzoku:
								value_list.append('金属系')
							elif i in value_borad:
								value_list.append('石膏ボード系')
							elif i in value_haipura:
								value_list.append('廃プラスチック系')
							else:
								value_list.append('その他')
						df2['分類'] = value_list

						type_list = [i for i in df2['集計区分'].unique()]
						select_type = st.sidebar.select_slider('種類を選択してください', type_list)
						df2 = df2[df2['集計区分'] == select_type]
						select_df = df2.groupby('得意先').sum()
						select_df = select_df.sort_values(sel_col, ascending=False)
						type_df = df2.groupby('分類').sum()
						type_df['分類'] = type_df.index
						fig = px.pie(type_df, values=sel_col, names='分類',title=f'{select_type}:{sel_col}構成比')
						st.plotly_chart(fig)
						product = st.select_slider('選択してください', type_df.index)
						product_df = df2[df2['分類'] == product]
						product_df = product_df.groupby('商品').sum()
						st.bar_chart(product_df[sel_col])
						st.write(f'{select_type} 上位順')
						st.write(select_df[colum_list])
						
						custmer = st.text_input('顧客名を入力してください')
						custm_list = []
						for i, v in enumerate(custmers):
							if custmer in v:
								custm_list.append(v)		
						chose = st.selectbox('顧客を選択してください ↓',custm_list)
						df2 = df2[df2['得意先'] == chose]
						month_list = [i for i in df2['month'].unique()]
							
						df2_valance = df2.groupby('集計区分').sum()
						df2_valance['集計区分'] = df2_valance.index
						df2_sum = df2.groupby('商品').sum()
						df2_sum['商品'] = df2_sum.index
						df2_day = df2.groupby('day').sum()
						df2_day['day'] = df2_day.index
						b = st.checkbox('data')
						if b == True:
							if sel_col == colum_list[0]:
								product_list = [i for i in df2_sum['商品'].unique()]
								options = st.sidebar.multiselect('商品を選択してください',product_list, product_list)
								df2_sum = df2_sum.loc[options]
								
								value = '正味重量_明細'
								st.write(df2_sum.iloc[:,:2])
								st.bar_chart(df2_sum.iloc[:,0])
								fig = px.pie(df2_sum, values=value, names='商品')
								fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
								st.plotly_chart(fig)
								data_check = st.checkbox('check')
								if data_check == True:
									if len(month_list) == 1:
										st.write(f'{month_list[0]}月 日量推移')
										st.bar_chart(df2_day[sel_col])
										sum_kg = df2_day[value].sum()
										st.write(f'{month_list[0]}月 合計{sum_kg}kg')
									else:
										choice_month = st.select_slider('選択してください', month_list, options=month_list[0])
										df2 = df2[df2['month'] == choice_month]
										df2_day = df2.groupby('day').sum()
										st.write(f'{choice_month}月 日量推移')
										st.bar_chart(df2_day[sel_col])
										sum_kg = df2_day[value].sum()
										st.write(f'{choice_month}月 合計{sum_kg}kg')
							
							else:
								value = '金額'
								if len(df2_sum['商品']) < 15:
									st.write(f'{chose}: データ全体 :商品構成比 {colum_list[1]}')	
									fig = px.pie(df2_sum, values=value, names='商品')
									st.plotly_chart(fig)
								else:
									st.bar_chart(df2_sum.loc[:,value])

								data_check = st.checkbox('check')
								if data_check == True:
									if len(month_list) == 1:
										st.write(f'{month_list[0]}月 売上金額推移')
										st.bar_chart(df2_day[sel_col])
										sum_kg = df2_day[value].sum()
										st.write(f'{month_list[0]}月 合計{sum_kg}円')
									else:
										choice_month = st.select_slider('選択してください', month_list, options=month_list[0])
										df2 = df2[df2['month'] == choice_month]
										df2_day = df2.groupby('day').sum()
										st.write(f'{choice_month}月 売上推移')
										st.bar_chart(df2_day[sel_col])
										sum_kg = df2_day[value].sum()
										st.write(f'{choice_month}月 合計{sum_kg}円')
						
			else:
				if file_3 != None:
					df3 = read_file(file_3)
					
					
					df3 = df3[['伝票番号','伝票日付','得意先','商品','集計区分','正味重量_明細','金額']]
					df3 = df3.dropna()
					con = sqlite3.connect('file3.db')
					cursor = con.cursor()
					create_data = """CREATE TABLE IF NOT EXISTS file3
							   (id INTEGER, day TEXT, custmers TEXT, product TEXT, type TEXT, kg INTEGER, revenue INTEGER)
							   """
					cursor.execute(create_data)
					insert_data = """
					              INSERT INTO file1 VALUES(?,?,?,?,?,?,?)
					              """
	
					for idx, row in df3.iterrows():
						cursor.execute(insert_data,(row))
					con.commit()
					custmers = [i for i in df3['得意先'].unique()] 
					df3['year'] = df3['伝票日付'].apply(lambda x: int(x[:4]))	
					df3['month'] = df3['伝票日付'].apply(lambda x: int(x.split('-')[1]))
					df3['day'] = df3['伝票日付'].apply(lambda x: int(x.split('-')[2]))


					button = st.checkbox('show')

					if button == True:
						year = df3['year'].unique()
						month = df3['month'].unique()
						kg_all = df3['正味重量_明細'].sum()
						kg_mean = kg_all / len(df3['day'].unique())
						st.write(f'{year}年 {month}月')
						st.write(f'顧客数:{len(custmers)} 搬入量:{kg_all}kg  日量平均:{kg_mean:.2f}kg') 
						df3_bar = df3.groupby('day').sum()
						st.bar_chart(df3_bar[['正味重量_明細','金額']])
						colum_list = ['正味重量_明細', '金額']
						sel_col = st.select_slider('選択してください', colum_list)
						df3_all = df3.groupby('得意先').sum()
						df3_all['得意先'] = df3_all.index
						df3_all = df3_all.iloc[:,:2]
						st.write(f'{sel_col}上位順')
						df3_all = df3_all.sort_values(sel_col, ascending=False)
						st.area_chart(df3_all[sel_col].values)
						with st.container():
							col1, col2 = st.columns([1, 1])
						with col1:
							st.write(df3_all)
						with col2:
							df3_v = df3.groupby('集計区分').sum()
							st.bar_chart(df3_v[['正味重量_明細']])
						value_list = []
						
						for i in df3['商品'].values:
							if i in value_kongou:
								value_list.append('混合系')
							elif i in value_keiryou:
								value_list.append('軽量系')
							elif i in value_shoukyaku:
								value_list.append('焼却系')
							elif i in value_kikuzu:
								value_list.append('木くず')
							elif i in value_kinzoku:
								value_list.append('金属系')
							elif i in value_borad:
								value_list.append('石膏ボード系')
							elif i in value_haipura:
								value_list.append('廃プラスチック系')
							else:
								value_list.append('その他')
						df3['分類'] = value_list

						type_list = [i for i in df3['集計区分'].unique()]
						select_type = st.sidebar.select_slider('種類を選択してください', type_list)
						df3 = df3[df3['集計区分'] == select_type]
						select_df = df3.groupby('得意先').sum()
						select_df = select_df.sort_values(sel_col, ascending=False)
						type_df = df3.groupby('分類').sum()
						type_df['分類'] = type_df.index
						fig = px.pie(type_df, values=sel_col, names='分類',title=f'{select_type}:{sel_col}構成比')
						st.plotly_chart(fig)
						product = st.select_slider('選択してください', type_df.index)
						product_df = df3[df3['分類'] == product]
						product_df = product_df.groupby('商品').sum()
						st.bar_chart(product_df[sel_col])
						st.write(f'{select_type} 上位順')
						st.write(select_df[colum_list])
						custmer = st.text_input('顧客名を入力してください')
						custm_list = []
						for i, v in enumerate(custmers):
							if custmer in v:
								custm_list.append(v)		
						chose = st.selectbox('顧客を選択してください ↓',custm_list)
						df3 = df3[df3['得意先'] == chose]
						month_list = [i for i in df3['month'].unique()]
							
						df3_valance = df3.groupby('集計区分').sum()
						df3_valance['集計区分'] = df3_valance.index
						df3_sum = df3.groupby('商品').sum()
						df3_sum['商品'] = df3_sum.index
						df3_day = df3.groupby('day').sum()
						df3_day['day'] = df3_day.index
						b = st.checkbox('data')
						if b == True:
							if sel_col == colum_list[0]:
								product_list = [i for i in df3_sum['商品'].unique()]
								options = st.sidebar.multiselect('商品を選択してください',product_list, product_list)
								df3_sum = df3_sum.loc[options]
								
								value = '正味重量_明細'
								st.write(df3_sum.iloc[:,:2])
								st.bar_chart(df3_sum.iloc[:,0])
								fig = px.pie(df3_sum, values=value, names='商品')
								fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
								st.plotly_chart(fig)
								data_check = st.checkbox('check')
								if data_check == True:
									if len(month_list) == 1:
										st.write(f'{month_list[0]}月 日量推移')
										st.bar_chart(df3_day[sel_col])
										sum_kg = df3_day[value].sum()
										st.write(f'{month_list[0]}月 合計{sum_kg}kg')
									else:
										choice_month = st.select_slider('選択してください', month_list, options=month_list[0])
										df3 = df3[df3['month'] == choice_month]
										df3_day = df3.groupby('day').sum()
										st.write(f'{choice_month}月 日量推移')
										st.bar_chart(df3_day[sel_col])
										sum_kg = df3_day[value].sum()
										st.write(f'{choice_month}月 合計{sum_kg}kg')
							
							else:
								value = '金額'
								if len(df3_sum['商品']) < 15:
									st.write(f'{chose}: データ全体 :商品構成比 {colum_list[1]}')	
									fig = px.pie(df3_sum, values=value, names='商品')
									st.plotly_chart(fig)
								else:
									st.bar_chart(df3_sum.loc[:,value])

								data_check = st.checkbox('check')
								if data_check == True:
									if len(month_list) == 1:
										st.write(f'{month_list[0]}月 売上金額推移')
										st.bar_chart(df3_day[sel_col])
										sum_kg = df3_day[value].sum()
										st.write(f'{month_list[0]}月 合計{sum_kg}円')

									else:
										choice_month = st.select_slider('選択してください', month_list, options=month_list[0])
										df3 = df3[df3['month'] == choice_month]
										df3_day = df3.groupby('day').sum()
										st.write(f'{choice_month}月 売上推移')
										st.bar_chart(df3_day[sel_col])
										sum_kg = df3_day[value].sum()
										st.write(f'{choice_month}月 合計{sum_kg}円')
								
								


