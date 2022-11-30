# %%
import pandas as pd
dfB = pd.read_csv('tweets.csv')


aux = dfB.groupby(["Local"])
aux.count()
aux.get_group("São Paulo, Brasil")
# %%
for i in  aux.groups.keys():
	print(i)
	aux.get_group(i).to_csv(i.replace(", ","-").replace(" ","_")+'.csv')
# %%