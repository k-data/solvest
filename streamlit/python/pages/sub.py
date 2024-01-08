import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
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
from sklearn.tree import DecisionTreeRegressor, plot_tree
import base64
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint, uniform


st.set_page_config(layout="wide")

def datetime(df, column):
	df[column] = pd.to_datetime(df[column])
	df['year_month'] = df[column].dt.to_period('M')

	df['年'] = df[column].dt.year
	df['月'] = df[column].dt.month
# ヒストグラムのプロット関数
def plot_histogram(data, category, title, lim):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='購入回数', hue='年', element='step', fill=False, common_norm=False, stat='density')
    plt.title(title)
    plt.xlabel('購入回数')
    plt.ylabel('密度')
    plt.xlim(0, lim)
    plt.xticks(np.arange(0, lim, lim*0.05))
    st.pyplot(plt)

# 累積確率のプロット関数
def plot_cumulative_distribution(data, category, title, lim):
    plt.figure(figsize=(10, 6))
    for year in [2021, 2022, 2023]:
        subset = data[data['年'] == year]
        ecdf = sns.ecdfplot(subset, x='購入回数', stat='proportion', label=str(year))
        plt.axhline(0.8, color='red', linestyle='--')
        y_values = ecdf.lines[year - 2021].get_ydata()  # 累積確率のy値
        x_values = ecdf.lines[year - 2021].get_xdata()  # 対応するx値
        x_intersect = next((x for x, y in zip(x_values, y_values) if y >= 0.8), None)

        if x_intersect is not None:
            plt.axvline(x=x_intersect, color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('購入回数')
    plt.ylabel('累積確率')
    plt.xlim(0, lim)
    plt.xticks(np.arange(0, lim, lim*0.05))
    plt.legend(title='Year')
    st.pyplot(plt)


if 'df_revenue' not in st.session_state:
	
	df3 = pd.read_csv('streamlit/data/2021_all.csv')
	df4 = pd.read_csv('streamlit/data/2022_all.csv')
	df5 = pd.read_csv('streamlit/data/2023_all.csv')

	
	datetime(df3, '伝票日付')
	datetime(df4, '伝票日付')
	datetime(df5, '伝票日付')

	st.session_state.df_revenue = pd.concat([df3, df4, df5])

df = pd.read_csv('streamlit/data/cost_all.csv')
df2 = pd.read_csv('streamlit/data/搬入集計.csv')
datetime(df, '伝票日付')
datetime(df2, '日付')
st.session_state.df_revenue = st.session_state.df_revenue.sort_values('伝票日付', ascending=True)
df_revenue_two =  st.session_state.df_revenue.groupby('集計区分').sum()['正味重量_明細']
df_revenue_type = st.session_state.df_revenue[st.session_state.df_revenue['集計区分'] == '処分費'].groupby('商品種別').sum()[['正味重量_明細', '金額']] 
df_revenue_product = st.session_state.df_revenue[(st.session_state.df_revenue['集計区分'] == '処分費') & (st.session_state.df_revenue['商品種別'] == '混合廃棄物')].groupby('商品').sum()[['正味重量_明細', '金額']]
df_dust = st.session_state.df_revenue[st.session_state.df_revenue['集計区分'] == '処分費'] 
type_list = list(df_revenue_type.index)
product_list = list(df_revenue_product.index)

st.subheader('1, 2021/01～2023/12までのオネスト搬入物の内訳')
col1, col2, col3 = st.columns(3)
select_options = ('正味重量_明細', '金額')
select_button = st.sidebar.radio('1, 選択してください', select_options)
st.sidebar.markdown('---')

with col1:

	ex1 = st.expander('搬入物の内訳')
	with ex1:
		df_revenue_two = df_revenue_two.iloc[:-1,]

		labels = ['処分費', '有価物']
		values = df_revenue_two.values

		fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
		st.plotly_chart(fig)
		st.write(df_revenue_two)

with col2:
	ex2 = st.expander('産廃の内訳')
	with ex2:
		labels = type_list
		values = df_revenue_type[select_button]
		fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
		st.plotly_chart(fig)
		st.write(df_revenue_type)


with col3:
	ex3 = st.expander('混合廃棄物の内訳')
	with ex3:
		labels = product_list
		values = df_revenue_product[select_button]
		fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
		st.plotly_chart(fig)
		st.write(df_revenue_product)

st.markdown('---')
date_select = st.sidebar.radio('2, 関係性を観る期間を選択してください', ('全体', '月'))
st.sidebar.markdown('---')

st.subheader('2, SOLVEST向けの荷物と、混合廃棄物の関係性')

if date_select == '全体':
	idx = '伝票日付'
else:
	idx = 'year_month'

if select_button == select_options[0]:
	unit = 'kg'
	value = '正味重量_明細'
else:
	unit = '円'
	value = '合計金額'

df_dust = pd.pivot_table(df_dust, index=idx, columns='商品種別', values=select_button, aggfunc='sum')
df_dust_type = st.session_state.df_revenue[st.session_state.df_revenue['商品種別'] == '混合廃棄物']
df_dust_type = pd.pivot_table(df_dust_type, index=idx, columns='商品', values=select_button, aggfunc='sum')

df_cost = df[df['新工場稼働後'] == 'SOLVEST']
df_cost = df.groupby(idx).sum()[[value]]

df_dust.fillna(0, inplace=True)
df_dust_type.fillna(0, inplace=True)

df_concat = pd.concat([df_dust, df_cost], axis=1)
df_concat_type = pd.concat([df_dust_type, df_cost], axis=1)
df_concat.fillna(0, inplace=True)
df_concat_type.fillna(0, inplace=True)



col4, col5 = st.columns(2)

with col4:
	ex4 = st.expander(f'商品カテゴリーとSOLVEST向けの荷物の相関関係')
	with ex4:
		st.write(f'単位: {unit}')
		fig, ax = plt.subplots()
		sns.scatterplot(data=df_concat_type, x='混合廃棄物', y='廃ﾌﾟﾗｽﾁｯｸ類', hue=value, size=value, ax=ax)
		plt.legend()
		st.pyplot(fig)
		
		st.write('相関係数')
		st.write(df_concat.corr().loc[value].sort_values(ascending=False)[:-1])

with col5:
	ex5 = st.expander(f'混合廃棄物カテゴリーとSOLVEST向け荷物の相関関係')
	with ex5:
		st.write(f'単位: {unit}')
		fig, ax = plt.subplots()
		sns.scatterplot(data=df_concat_type, x='混合廃棄物Ａ', y='混合廃棄物Ｂ', hue=value, size=value, ax=ax)
		plt.legend()
		st.pyplot(fig)
		st.write('相関係数')
		st.write(df_concat_type.corr().loc[value].sort_values(ascending=False)[:10])

st.markdown('---')
customer_purchase_counts = st.session_state.df_revenue.groupby(['年', '得意先']).size().reset_index(name='購入回数')

# 商品ごとの購入回数の計算
product_purchase_counts = st.session_state.df_revenue.groupby(['年', '商品']).size().reset_index(name='購入回数')

ex6 = st.expander('得意先ごとの購入回数')
with ex6:
	st.subheader('得意先ごとの購入回数のヒストグラムと累積確率分布')

	col6, col7 = st.columns(2)
	with col6:
		plot_histogram(customer_purchase_counts, '得意先', '得意先ごとの購入回数のヒストグラム', lim=800)
		st.write('年度別')
		st.write(customer_purchase_counts)
	with col7:
		plot_cumulative_distribution(customer_purchase_counts, '得意先', '得意先ごとの購入回数の累積確率分布', lim=1000)
		custom_sum = customer_purchase_counts.groupby('得意先').sum()[['購入回数']]
		st.write('3年合計')
		st.write(custom_sum)

st.markdown('---')
ex7 = st.expander('商品ごとの購入回数')
with ex7:
	st.subheader('商品ごとの購入回数のヒストグラムと累積確率分布')
	col8, col9 = st.columns(2)

	with col8:
		plot_histogram(product_purchase_counts, '商品', '商品ごとの購入回数のヒストグラム', lim=900)
		st.write('年度別')
		st.write(product_purchase_counts)
	with col9:
		plot_cumulative_distribution(product_purchase_counts, '商品', '商品ごとの購入回数の累積確率分布', lim=10000)
		product_sum = product_purchase_counts.groupby('商品').sum()[['購入回数']]
		st.write('3年合計')
		st.write(product_sum)








