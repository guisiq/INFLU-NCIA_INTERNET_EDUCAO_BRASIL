
# %% [markdown]
# conectando a uma pasta no drive

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %% [markdown]
# importando as bibliotecas nessesarias 

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %% [markdown]
# carregando os dados exemplo
# 

# %% [markdown]
# caregando os dados pratica 

# %%
# lendo dados do csv para um data frame
# df = pd.read_csv('/content/drive/MyDrive/db.csv')
df = pd.read_csv('C:\\Users\\lopes\\Documents\\facudade\\estatistica\\trabalho de estatistica\\db.csv')
# display(df)

# %%
df.isnull().any()

# %%
# df = df.where(df['densidade'] != np.NAN)
# df = df.where(df['taxa_aprovacao'] != np.NAN)
# df = df.where(df['densidade'] != np.Infinity)
# df = df.where(df['taxa_aprovacao'] != np.Infinity)


# %%
y = df['taxa_aprovacao']
X = df['densidade']

# %%
print(y)
print(X)

# %%
# transformX = Normalizer().fit([X])
# transformY = Normalizer().fit([y])
# display(transformX)
# display(transformY)

# %%
X = Normalizer().fit([X]).transform([X])
y = Normalizer().fit([y]).transform([y])

# %%
# display(X.reshape(-1,1))
# display(X)
# display(y)

# %% [markdown]
# Selecionando Variáveis para o modelo
# 

# %% [markdown]
# 

# %%
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1,1), y.reshape(-1,1), test_size=0.25)


# %%
# display(X_train, X_test, y_train, y_test)

# %% [markdown]
# 

# %%
# Criando o modelo LinearRegression
regr = LinearRegression().fit(X_train, y_train)
regr2 = LinearRegression().fit(X, y)

# %%

# Realizar predição com os dados separados para teste
y_pred = regr.predict(X_test)
y_pred2 = regr2.predict(X)
# Visualização dos 20 primeiros resultados
y_pred[:20]
# %%
plt.scatter(X_test, y_test,  color='black')
plt.scatter(X_train, y_train,  color='gray')
plt.plot(X_test, y_pred   , color='blue', linewidth=3)
plt.plot(X, y_pred2  , color='red', linewidth=2)
plt.show()