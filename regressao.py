
# %% [markdown]
# conectando a uma pasta no drive
# from google.colab import drive
# drive.mount('/content/drive')

# %% [markdown]
# importando as bibliotecas nessesarias 

# %%
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#%%
# %%
def Regressao(X , y , imprimirMetricas = False ,tituloGrafico = False,axis = ('x','y')):
	X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1,1), y.reshape(-1,1), test_size=0.25)

	regr = LinearRegression().fit(X_train, y_train)

	# Realizar predição com os dados separados para teste
	y_pred = regr.predict(X_test)

	# salvando metricas 
	# metricas = {
	#         'Mean squared error:': mean_squared_error(y_test, y_pred),
	#         'R2 Score:' : r2_score(y_test, y_pred),
	#         'MAE:':  mean_absolute_error(y_test, y_pred),
	#         'score': regr.score(X_test,y_test),
	#         'intercept_': regr.intercept_,
	#         'coef_' :regr.coef_[0][0],
	#         'funcao':'y = '+str(regr.intercept_[0])+'+('+str(regr.coef_[0][0])+'x)'
	#         }
	metricas = [
			['Mean squared error:',mean_squared_error(y_test, y_pred)],
			['R2 Score:' , r2_score(y_test, y_pred)],
			['MAE:',  mean_absolute_error(y_test, y_pred)],
			['score', regr.score(X_test,y_test)],
			['intercept_', regr.intercept_],
			['coef_' ,regr.coef_[0][0]],
			['funcao','y = '+str(regr.intercept_[0])+'+('+str(regr.coef_[0][0])+'x)']
		]
	
	if imprimirMetricas == True:
		print(metricas)
	# print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
	# print('R2 Score: %.2f' % r2_score(y_test, y_pred))
	# print('MAE: %.2f' % mean_absolute_error(y_test, y_pred))
	# print('regr.score(X_train,y_train): %.2f' % regr.score(X_test,y_test))
	if tituloGrafico != False:
		
		plt.scatter(X, y,  color='black')
		plt.plot(X_test, y_pred   , color='blue', linewidth=3)
		
		plt.xlabel(axis[0])
		plt.ylabel(axis[1])
		# plt.text(str(metricas))
		# anchored_text = AnchoredText(str(metricas), loc=2)
		# plt.ax.add_artist(anchored_text)
		plt.show()
		plt.savefig(tituloGrafico,transparent = True)
	return metricas
# %%
def violinePlot(df,qtClasses,
				tituloGrafico = False,
				axis = ('x','y'),
				xLabel = "densidade",
				yLabel = "taxa_aprovacao"):
	df = df[np.isnan(df[yLabel]) !=True ]  
	df = df[np.isnan(df[xLabel]) !=True ]  
	dataset = []
	intervaloClasse = ((X.max()-X.min()) /qtClasses)
	minClass = 0.0 
	maxClasse = intervaloClasse
	arr = df[(df[xLabel] < maxClasse) & (df[xLabel] > minClass)][yLabel].values
	pos = []
	pos.append(maxClasse)
	print(arr)

	dataset.append(arr)
	for i in range(qtClasses):
		minClass = maxClasse 
		maxClasse += intervaloClasse
		arr = df[(df[xLabel] < maxClasse) & (df[xLabel] > minClass)][yLabel].values
		if arr.__len__()!= 0:
			pos.append(maxClasse)
			dataset.append(arr)
	# plotando o grafico
	fig, ax = plt.subplots(figsize=(13,10))


	ax.yaxis.grid(True)
	# ax.violinplot(dataset=dataset)
	ax.violinplot(dataset, pos,  points=200, widths=4,
						showmeans=True, showextrema=True, showmedians=True)
	plt.xlabel(axis[0])
	plt.ylabel(axis[1])
	if tituloGrafico != False:
		plt.title = tituloGrafico 
		plt.show()
		plt.savefig(tituloGrafico,transparent = True)
# %% [markdown]
# lendo dados do csv para um data frame
# %%
# df = pd.read_csv('/content/drive/MyDrive/db.csv')
dfB = pd.read_csv('db2.csv')
df = dfB[np.isnan(dfB["taxa_aprovacao"]) !=True ]
df.isnull().any()
# %%


# %%
y = df['taxa_aprovacao']
X = df['densidade']
# criando um violin plots
# %%
# separando dados em classes




# %% imprimindo distribuicao normal da taxa_aprovacao 
# %%

plt.hist(X, 300) 
plt.show() 

plt.hist(y, 300) 
plt.show()
# %%
X = Normalizer().fit([X]).transform([X])
y = Normalizer().fit([y]).transform([y])

# %% 
# Selecionando Variáveis para o modelo
metricasNormalizadas = Regressao(X,y,True,'taxa aprovação X densidade de conexao normalizado',('taxa aprovação','densidade de conexao'))
y = df[['taxa_aprovacao']].to_numpy()
X = df[['densidade']].to_numpy()
metricasNaoNormalizadas = Regressao(X,y,True,'taxa aprovação X densidade de conexao',('taxa aprovação','densidade de conexao'))
violinePlot(df = dfB,
			qtClasses=30,
			tituloGrafico="taxa aprovação X densidade de conexao distribuicao",axis=("densidade de conexao","taxa aprovação ")
			)
# violinePlot(df = df,
# 			qtClasses = 10,
# 			tituloGrafico = "taxa aprovação X densidade de conexao distribuicao",axis=("densidade de conexao","taxa aprovação ")
# 			xlabel = "")