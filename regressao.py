
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
		# plt.show()
		plt.savefig(tituloGrafico,transparent = True)
	return metricas
# %%

def pontoMedio(x,y):
	return (x+y)/2
def violinePlot(df,
				qtClasses,
				imprimirMetricas = False,
				tituloGrafico = False,
				axis = ('densidade','taxa aprovacao'),
				xLabel = "densidade",
				yLabel = "taxa_aprovacao",
				normalizar = False):
	
	if(tituloGrafico == True):	
		tituloGrafico = axis[0] +" X "+axis[1]
	
	df = df[(np.isnan(df[yLabel]) !=True) & (np.isnan(df[xLabel]) !=True) ]  
	
	# df = df[ ]  
	y = df[yLabel].to_numpy()
	X = df[xLabel].to_numpy()
	if(np.count_nonzero(y)<1 or np.count_nonzero(X)<1):
		return {
			'Mean squared error:':None,
			'R2 Score:' :None,
			'MAE:':None,
			'score': None,
			'intercept_':None,
			'coef_':None,
			'funcao':None,
			'relacao':yLabel
	}
	widths=5
	if(normalizar == True):
		y = Normalizer().fit([y]).transform([y])[0]
		X = Normalizer().fit([X]).transform([X])[0]
		df = pd.DataFrame({
			xLabel:X,
			yLabel:y
		})
		widths=0.004
	# y = df[xLabel].to_numpy()
	# X = df[yLabel].to_numpy()
	
	#regressao


	X = X.reshape(-1,1)
	y = y.reshape(-1,1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
	regr = LinearRegression().fit(X_train, y_train)

	# Realizar predição com os dados separados para teste
	y_pred = regr.predict(X_train)
	metricas = {
			'Mean squared error:':mean_squared_error(y_train, y_pred),
			'R2 Score:' : r2_score(y_train, y_pred),
			'MAE:':  mean_absolute_error(y_train, y_pred),
			'score': regr.score(X,y),
			'intercept_': regr.intercept_,
			'coef_': regr.coef_[0][0],
			'funcao':'y = '+str(regr.intercept_[0])+'+('+str(regr.coef_[0][0])+'x)',
			'relacao':yLabel
	}
	
		
	if imprimirMetricas == True:
		print(metricas)
	if tituloGrafico != False:
		dataset = []
		intervaloClasse = ((X.max()-X.min()) /qtClasses)
		minClass = 0.0 
		maxClasse = intervaloClasse

		arr = df[(df[xLabel] < maxClasse) & (df[xLabel] > minClass)][yLabel].values
		pos = []
		if arr.__len__()!= 0:
			pos.append(pontoMedio(minClass, maxClasse))
			dataset.append(arr)
		for i in range(qtClasses):
			minClass = maxClasse 
			maxClasse += intervaloClasse
			arr = df[(df[xLabel] < maxClasse) & (df[xLabel] > minClass)][yLabel].values
			if arr.__len__()!= 0:
				pos.append(pontoMedio(minClass, maxClasse))
				dataset.append(arr)
		# plotando o grafico
		
		fig, ax = plt.subplots(figsize=(13,10))


		ax.yaxis.grid(True)
		ax.set_xlabel(axis[0])
		ax.set_ylabel(axis[1])
		ax.set_ylim([0, 1.07*y.max()])
		ax.set_xlim([0, 1.1*X.max()])
		ax.violinplot(dataset, 
							pos,  
							points=200, 
							widths=widths,
							showmeans=True, 
							showextrema=True, 
							showmedians=True)
		ax.plot(X_train, y_pred   , color='blue', linewidth=3)
		plt.title(tituloGrafico) 
		plt.savefig(tituloGrafico ,transparent = True)
		# plt.close(fig)
		# plt.show()
	return metricas
# %% [markdown]
# lendo dados do csv para um data frame
# %%
dfB = pd.read_csv('db2.csv')

# %%
def compararDensidadeConexao(
	df,
	yLabels=["indicador_rendimento","taxa_aprovacao","nota_saeb_media_padronizada","nota_saeb_lingua_portuguesa","nota_saeb_matematica",],
	filtrosRealizados = "",
	imprimir = True
):
	dadosRegressao = []
	for i in yLabels:
		if(imprimir):
			imprimir = "("+str(filtrosRealizados)+")" + "densidade de conexao X "+i.replace("_"," ")
		aux = violinePlot(df = df,
				qtClasses = 25,
				tituloGrafico = imprimir,
				axis=("densidade de conexao",i.replace("_"," ")),
				yLabel = i,
				normalizar = True)
		if(filtrosRealizados != ""):
			aux['filtro'] = filtrosRealizados
		dadosRegressao.append(aux) 
	return dadosRegressao
# %%
dadosRegressao = []
dadosRegressao.extend(compararDensidadeConexao(dfB))

grupos = dfB.groupby(["sigla_uf"])
grupos.count()
for i in  grupos.groups.keys():
	dadosRegressao.extend(compararDensidadeConexao(grupos.get_group(i),filtrosRealizados = "sigla_uf-"+i))
	print(i)
grupos = dfB.groupby(["ano"])
grupos.count()
for i in  grupos.groups.keys():
	dadosRegressao.extend(compararDensidadeConexao(grupos.get_group(i),filtrosRealizados = "Ano-"+str(i)))
	print(i)
dt = pd.DataFrame(dadosRegressao)
dt.to_csv("dadosRegressao.csv")

#%%
Regressao([dfB])
dt.to_json("dadosRegressao.json",orient="records")	
dt.to_excel("dadosRegressao.xlsx")