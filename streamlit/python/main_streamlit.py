import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import pydeck as pdk
import plotly.express as px 
import streamlit as st
st.set_page_config(layout="wide")
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
from sklearn.tree import DecisionTreeRegressor, plot_tree
import base64
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint, uniform


def datetime(df, column):
	df[column] = pd.to_datetime(df[column])
	df['year_month'] = df[column].dt.to_period('M')

	df['年'] = df[column].dt.year
	df['年'] = df['年'].apply(lambda x: int(x))
	df['月'] = df[column].dt.month
	df['月'] = df['月'].apply(lambda x: int(x))
def on_button_click_1():
    st.session_state['button_clicked_1'] = not st.session_state['button_clicked_1']

# ボタン2が押されたときの動作
def on_button_click_2():
    st.session_state['button_clicked_2'] = not st.session_state['button_clicked_2']

df = pd.read_csv('streamlit/data/cost_all.csv')
df2 = pd.read_csv('streamlit/data/搬入集計.csv')
df3 = pd.read_csv('streamlit/data/2021_all.csv')
df4 = pd.read_csv('streamlit/data/2022_all.csv')
df5 = pd.read_csv('streamlit/data/2023_all.csv')

datetime(df, '伝票日付')
datetime(df2, '日付')
datetime(df3, '伝票日付')
datetime(df4, '伝票日付')
datetime(df5, '伝票日付')


df_year = df.groupby(['年', '新工場稼働後']).sum()[['正味重量_明細', '合計金額']]
df2_year = df2.groupby('年').sum()

col1, col2, col3 = st.columns(3)

with col1:
	st.write('年別工場実績')
	st.write(df2_year[['台数', '数量', '工場売上', '工場粗利', '工場仕入']])
	df2_year = df2_year[['台数', '数量', '工場売上', '工場粗利', '工場仕入']]
with col2:
	st.write('年別SOLVEST向け実績')
	st.write(df_year)

with col3:
	df_sample = pd.DataFrame(columns=['数量比(%)', '金額比(%)'], index=[2021, 2022, 2023])
	df_year_type = df_year.reset_index()
	df_year_type = df_year_type[df_year_type['新工場稼働後'] == 'SOLVEST']
	kg_2021 = df_year_type[df_year_type['年'] == 2021]['正味重量_明細'] / df2_year.loc[2021, '数量']
	kg_2021 *= 100
	kg_2021 = np.array(kg_2021)
	kg_2022 = df_year_type[df_year_type['年'] == 2022]['正味重量_明細'] / df2_year.loc[2022, '数量']
	kg_2022 *= 100
	kg_2022 = np.array(kg_2022)
	kg_2023 = df_year_type[df_year_type['年'] == 2023]['正味重量_明細'] / df2_year.loc[2023, '数量']
	kg_2023 *= 100
	kg_2023 = np.array(kg_2023)
	kg_all = df_year_type['正味重量_明細'].sum() / df2_year['数量'].sum()
	kg_all *= 100
	kg_all = np.array(kg_all)
	cost_2021 = df_year_type[df_year_type['年'] == 2021]['合計金額'] / df2_year.loc[2021, '工場仕入']
	cost_2021 *= 100
	cost_2021 = np.array(cost_2021)
	cost_2022 = df_year_type[df_year_type['年'] == 2022]['合計金額'] / df2_year.loc[2022, '工場仕入']
	cost_2022 *= 100
	cost_2022 = np.array(cost_2022)
	cost_2023 = df_year_type[df_year_type['年'] == 2023]['合計金額'] / df2_year.loc[2023, '工場仕入']
	cost_2023 *= 100
	cost_2023 = np.array(cost_2023)
	cost_all = df_year_type['合計金額'].sum() / df2_year['工場仕入'].sum()
	cost_all *= 100
	cost_all = np.array(cost_all)
	df_sample.loc[2021, '数量比(%)'] = np.round(kg_2021, 2)
	df_sample.loc[2022, '数量比(%)'] = np.round(kg_2022, 2)
	df_sample.loc[2023, '数量比(%)'] = np.round(kg_2023, 2)
	df_sample.loc[2021, '金額比(%)'] = np.round(cost_2021, 2)
	df_sample.loc[2022, '金額比(%)'] = np.round(cost_2022, 2)
	df_sample.loc[2023, '金額比(%)'] = np.round(cost_2023, 2)

	st.write('SOLVEST 比率')
	st.write(df_sample)
	st.write(f'全体数量比:{np.round(kg_all, 2)}%, 全体金額比: {np.round(cost_all, 2)}%')


expander = st.expander('全体比較')
with expander:
	options = ["金額", "数量"]
	selected_option = st.radio("どちらか選択してください", options=options)
	if selected_option == options[0]:
		df_plot = df.copy()
		
		df_plot = df_plot.groupby(['year_month', '新工場稼働後']).sum()[['正味重量_明細', '合計金額']]
		df_plot.reset_index(inplace=True)




		# '新工場稼働後' 列をピボットし、新しいデータフレームを作成
		df_pivot = df_plot.pivot(index='year_month', columns='新工場稼働後', values='合計金額')
		df_pivot_mean = df_pivot['SOLVEST'].rolling(window=6).mean()
		df_pivot_mean = pd.DataFrame(df_pivot_mean, index=df_pivot.index)
		df_pivot_mean.dropna(inplace=True)
		x2 = np.arange(5,len(df_pivot))
		fig, ax = plt.subplots()

		# 積み上げ棒グラフの作成
		df_pivot.plot(kind='bar', stacked=True, ax=ax)
		df2['工場仕入'].plot(kind='line', color='green', style='--', label='工場仕入')
		plt.plot(x2, df_pivot_mean['SOLVEST'], 'r-', label='SOLVEST: 6カ月移動平均')
		ax.set_title('合計金額の比較', fontsize=16)
		ax.set_xlabel('日付', fontsize=12)
		ax.set_ylabel('合計金額', fontsize=12)

		# 軸のフォーマット調整
		ax.tick_params(axis='x', rotation=90)
		ax.tick_params(axis='y', labelsize=10)

		# グリッド線のスタイル調整
		ax.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

		# 凡例の位置調整
		ax.legend(loc='upper right')

		# Streamlitへのグラフの表示
		st.pyplot(fig)
	else: 
		df_plot = df.copy()
		
		df_plot = df_plot.groupby(['year_month', '新工場稼働後']).sum()[['正味重量_明細', '合計金額']]
		df_plot.reset_index(inplace=True)




		# '新工場稼働後' 列をピボットし、新しいデータフレームを作成
		df_pivot = df_plot.pivot(index='year_month', columns='新工場稼働後', values='正味重量_明細')
		df_pivot_mean = df_pivot['SOLVEST'].rolling(window=6).mean()
		df_pivot_mean = pd.DataFrame(df_pivot_mean, index=df_pivot.index)
		df_pivot_mean.dropna(inplace=True)
		x2 = np.arange(5,len(df_pivot))
		fig, ax = plt.subplots()

		# 積み上げ棒グラフの作成
		df_pivot.plot(kind='bar', stacked=True, ax=ax)
		df2['数量'].plot(kind='line', color='green', label='搬入数量')
		plt.plot(x2, df_pivot_mean['SOLVEST'], color='red', label='SOLVEST: 6カ月移動平均')
		ax.set_title('出来高重量の比較', fontsize=16)
		ax.set_xlabel('日付', fontsize=12)
		ax.set_ylabel('重量', fontsize=12)

		# 軸のフォーマット調整
		ax.tick_params(axis='x', rotation=90)
		ax.tick_params(axis='y', labelsize=10)

		# グリッド線のスタイル調整
		ax.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

		# 凡例の位置調整
		ax.legend(loc='upper right')

		# Streamlitへのグラフの表示
		st.pyplot(fig)











expander2 = st.expander('SOLVEST向け構成比較')
with expander2:
	df_solvest = df[df['新工場稼働後'] == 'SOLVEST']
	df_cost = df_solvest.groupby('年').sum()[['金額', '運搬費', '合計金額']]
	col1, col2 = st.columns(2)
	if 'button_clicked_1' not in st.session_state:
		st.session_state['button_clicked_1'] = False

	if 'button_clicked_2' not in st.session_state:
		st.session_state['button_clicked_2'] = False

# ボタン1が押されたときの動作


	# 最初の列にデータフレームを表示
	with col1:
		st.write("データフレーム:")
		st.write(df_cost)
			
	# 二番目の列にデータフレームの記述統計を表示
	with col2:
		st.write("df_cost の記述統計:")
		st.write(df_cost.describe())
	
	button_r_1 = st.button('plot_Button', on_click=on_button_click_1)
	if st.session_state['button_clicked_1']:
		year_select = st.radio('年を選択してください', (2021, 2022, 2023))

		col3, col4 = st.columns(2)

		with col3:
			df_cost_year = pd.DataFrame(df_cost.loc[year_select, ['金額', '運搬費']])
			st.write(f'{year_select}年 SOLVEST向け金額構成比')
			labels = ['金額', '運搬費']
			values = df_cost_year[year_select]
			fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
			st.plotly_chart(fig)

		with col4:
			options = ['運搬費', '金額']
			select_value = st.radio('選択してください', options)
			df_solvest_year = df_solvest[df_solvest['年'] == year_select]
			df_solvest_year = df_solvest_year.groupby('種類').sum()[['正味重量_明細', '運搬費', '金額','合計金額']]
			df_solvest_year.reset_index(inplace=True)
			# outer_labels = ['運搬費', '']
			labels = df_solvest_year['種類']
			values = df_solvest_year[select_value]
			fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
			st.plotly_chart(fig)








