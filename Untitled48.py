#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install numpy pandas matplotlib seaborn scikit-learn folium plotly dash jupyterlab


# In[2]:


import pandas as pd
from sklearn.datasets import load_iris

# データをロード
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# データの初期処理
iris_df['species'] = iris_df['target'].map(dict(enumerate(iris.target_names)))
iris_df.drop(columns='target', inplace=True)


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt

# 統計的要約
print(iris_df.describe())

# ペアプロットでの視覚化
sns.pairplot(iris_df, hue='species')
plt.show()


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop('species', axis=1), iris_df['species'], test_size=0.2, random_state=42)

# モデルをトレーニング
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# テストデータでの評価
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop('species', axis=1), iris_df['species'], test_size=0.2, random_state=42)

# モデルをトレーニング
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# テストデータでの評価
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


# In[6]:


# データのクリーニングと前処理のデモンストレーションコード
iris_df.dropna(inplace=True)  # 欠損値の削除
print("欠損値の処理後のデータ概要:")
print(iris_df.info())


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

# ペアプロットの作成
sns.pairplot(iris_df, hue='species')
plt.show()


# In[8]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# モデルのパラメータ調整
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
print("Best parameters:", clf.best_params_)


# In[9]:


# 箱ひげ図の作成
sns.boxplot(x='species', y='petal length (cm)', data=iris_df)
plt.show()


# In[13]:


get_ipython().system('pip install pandasql')


# In[17]:


import pandas as pd
from sklearn.datasets import load_iris
import pandasql as ps

# データをロードしてデータフレームを作成
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
iris_df['species'] = iris.target_names[iris.target]

# データフレームの列名を確認
print("Dataframe columns:\n", iris_df.columns)

# SQLクエリを実行する関数
def execute_sql(query, local_env):
    return ps.sqldf(query, local_env)

# SQLクエリの定義
query = """
SELECT species, AVG(sepal_length) as avg_sepal_length
FROM iris_df
GROUP BY species
"""

# クエリの実行と結果の表示
result = execute_sql(query, locals())
print("Query result:\n", result)


# In[19]:


import folium

# 地図の作成
map = folium.Map(location=[45.5236, -122.6750], zoom_start=13)  # 例: ポートランド
folium.Marker([45.5236, -122.6750], popup='Portland').add_to(map)

# Jupyter Notebookで地図を表示
map



# In[21]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Dashアプリの基本的な構造
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='graph'),
    dcc.Dropdown(
        id='species-selector',
        options=[{'label': i, 'value': i} for i in iris_df['species'].unique()],
        value='setosa'
    )
])

@app.callback(
    Output('graph', 'figure'),
    [Input('species-selector', 'value')]
)
def update_graph(selected_species):
    filtered_df = iris_df[iris_df['species'] == selected_species]
    fig = px.scatter(filtered_df, x='sepal width (cm)', y='sepal length (cm)', color='species')
    return fig

# アプリを異なるポートで実行する
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)


# In[22]:


from sklearn.metrics import classification_report

# モデルの評価
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))


# In[ ]:




