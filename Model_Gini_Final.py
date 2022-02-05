#--python3.6.7
#=========GINI INDEX CODE=====
# Alejandro Duran - 
#07.05.2021

##SUMMARY: Analize economic indicators (from the World Bank database) and their influence 
## with respect to the Gini index (inequality measurement).
## This code is presented with several comments and it is intended to be self-explanatory

#====Import Libraries=====
import numpy as np									#basic library - maths
import pandas as pd									#data analysis and manipulation
import seaborn as sns  								 #data visualization - plots
import statsmodels.api as sm						#statistical models
from sklearn.metrics import mean_squared_error		#metrics and scoring
from sklearn.linear_model import LinearRegression	#For linear regression
from sklearn.ensemble import RandomForestRegressor  #random Forest 
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt						#data visualization
plt.rcParams.update({'font.size': 12})				#set global font size
#import sqlite3										# allows accessing the database using a nonstandard variant of the SQL query language

import wbdata	#<--									#I'm using this module from Oliver Sherouse (https://github.com/OliverSherouse/wbdata) 
													##This offers easy access to the World Bank Indicators (API ->  https://datahelpdesk.worldbank.org/knowledgebase/articles/889386)
													## Also check https://datacatalog.worldbank.org/     https://wbdata.readthedocs.io/en/stable/ 



#==========================================================================================
#							 Load Data and Data arrangement
#==========================================================================================

#====Load Data - World Bank Database=====
#wbdata.get_source()  #-> to check all indicators and its ID number ~el catálogo

#--Indicators to consider--
#SI.POV.GINI		->	GINI Index
#UIS.X.US.FSGOV  	->  Government expenditure on education, US$ (millions)
#FP.CPI.TOTL.ZG	 	->	Inflation, consumer prices (annual %)	-
#SL.UEM.TOTL.NE.ZS  ->	Unemployment, total (% of total labor force) (modeled ILO estimate)


#Selected Countries
#countries = ["USA","CHE","CO","VE","CL","ES","PT","DE","IT","FR","BR","BO","PA","NI","GB"] 
countries = ["CHE","ES","PT","DE","IT","FR","NL","GB","NO","FI","BE","DK","SE"] #Western Europe

indicators = {"SI.POV.GINI	": "GINI Index", "UIS.X.US.FSGOV": "Gov. Education exp US$ (millions) ","FP.CPI.TOTL.ZG": "Inflation",
		 "SL.UEM.TOTL.NE.ZS":"Unemployment(% of total labor force)"}

#Load ->data frame
df = wbdata.get_dataframe(indicators, country=countries, convert_date=False)

#Check NaN values per column
#df.isnull().sum(axis=0)
#dfn=df.dropna()  #Only consider rows/years where all indicators are available 

#-Save to .CVS (if needed)
#path="/Users/alejandro/Documents/Career_Paths/ILO/Code/test.csv"
# df.to_csv(path)


#Rearrange data frame for plotting
dfp = df.unstack(level=0)
dfn2=dfp.unstack(level=0)


#==========================================================================================
#							 First Analysis 
#==========================================================================================

#--Figure--   
#--Temporal evolution Gini index - All countries
dfp.plot(y="GINI Index")
plt.legend(loc='best')
plt.title("GINI Index (World Bank Data Base)")
plt.xlabel('Date'); plt.ylabel('GINI Index')
plt.ylim([25, 70]) #60
plt.grid()
plt.show()


#--Figure--   
#--- Density Distribution Gini Index---
sns.distplot(df['GINI Index'].dropna())
#sns.set_color_codes()
plt.show()


#--- Figure Correlations -----
df_corr=df.iloc[:,0:].corr()
mask = np.zeros(df_corr.shape, dtype=bool)
mask[np.tril_indices(len(mask))] = False
sns.heatmap(df_corr, annot = True, mask = mask);
plt.savefig('Correl_Europe.pdf')  
plt.show()

#--- Figure Pair crossplots -----
sns.pairplot(df.iloc[:,0:])
plt.show()


#--Figure--   
df.plot(x="Unemployment(% of total labor force)", y="GINI Index",kind="scatter")
plt.legend(loc='best'); 
plt.title("GINI Vs Unemployment - Western Europe"); 
plt.show()

#------------------------------
#------- Time Selection ---

# Date Range Selection: t1=[2003-2008]->Row[43-49]    AND  t2=[2009-2015]->[50-56] 
#dfp.iloc[43] 

#Only data with GINI Index attribute
dfg_t1=dfp["GINI Index"].values[43:49,:]					
dfg_t2=dfp["GINI Index"].values[51:56,:]				

#Only data with Unemployment attribute
dfu_t1=dfp["Unemployment(% of total labor force)"].values[43:49,:]
dfu_t2=dfp["Unemployment(% of total labor force)"].values[51:56,:]


#--Figure--
colors = ['k', 'r','g']
figt1 = plt.scatter(dfu_t1,dfg_t1,color=colors[0])  #Plot them - scatter
figt2 = plt.scatter(dfu_t2,dfg_t2,color=colors[1])
plt.legend((figt1, figt2),
           ('[2003-2008]', '[2009-2015]'),
           scatterpoints=1,
           loc='upper right',
          # ncol=3,
           fontsize=12)
plt.xlabel('Unemployment(% of total labor force)'); plt.ylabel('GINI Index');
plt.show()



#==========================================================================================
#								Models
#==========================================================================================

#-----------------------Model 1: Linear Regression --------------
#Arrange data
dfg_t1=np.asarray(dfg_t1).reshape(-1) 
dfg_t2=np.asarray(dfg_t2).reshape(-1) 

# Data selection - Chose Period t1 or t2
x=dfu_t2  
y=dfg_t2

#Do not consider NaN values
nanidx=np.argwhere(np.isnan(y))
x = np.delete(x, nanidx)
x=np.asarray(x).reshape((-1, 1))

nanask = np.isnan(y)
yc = ~ nanask
y = y[yc]

max_x =max(x) 
min_x =min(x) 

#--Linear Regression
model = LinearRegression()
model.fit(x, y)

#Performance 
ypred=model.predict(x)
mean_squared_error,r2_score,explained_variance_scoreprint 
r_sq = model.score(x, y) #R^2
("R2 :",r2_score(y,ypred))  #R^2  Again
print ("MSE= ",mean_squared_error(y,ypred))
print("RMSE= ",np.sqrt(mean_squared_error(y,ypred)))


# Parameters m=[slope , intercept with y-axis ]
m1=[model.coef_ ,model.intercept_] 
m2=[0.42678136, 27.06086279528636] #Parameters of the regression over data t2


# x1 = np.arange(min_x, max_x, 1)
# x2 =dfu_t2
# x2=np.sort(x2) #Another way to order it
x1 = np.arange(2, 25, 1)
x2=x1

#linear equation
y1 = m1[0] * x1 + m1[1] 
y2 = m2[0] * x2 + m2[1] 


#--Figure-- 
figt1=plt.scatter(x,y,color=colors[0]) #black
figt2=plt.scatter(dfu_t2,dfg_t2,color=colors[1]) #red
fit1=plt.plot(x1,y1, 'k') 
fit2=plt.plot(x2,y2, 'r') 
plt.legend((figt1, figt2, fit1, fit2),('[2003-2008]', '[2009-2015]')),
plt.xlabel('Unemployment(% of total labor force)'); plt.ylabel('GINI Index');
plt.show()




# OLR - lin regression as well to compare
# x=dfu_t1  #dfu_t2
# y=dfg_t1  #dfg_t2	
# model = smf.OLS(x,y)    
# results = model.fit()
# results.summary()
# m1=results.params #Parameters (Slope and coeff) of the regression



#--Figure-- 
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
#ax1.set_xlabel('Unemployment(%)') 
ax2.set_xlabel('Unemployment(%)') 
ax1.set_ylabel('Gini Index') 
ax2.set_ylabel('Gini Index') 
ax1.title.set_text('Period [2003-2008]')
ax2.title.set_text('Period [2009-2015]')
figt1=ax1.scatter(x,y,color=colors[0]) #black t1
fit1=ax1.plot(x1,y1, 'k') 
figt2=ax2.scatter(dfu_t2,dfg_t2,color=colors[1]) #red t2
fit2=ax2.plot(x2,y2, 'r') 
#plt.grid()
plt.show()


#=====================Model 2: Random Forest===================== 
#x.shape ->Dimension
xr=x.reshape(-1, 1)
yr=y.reshape(-1, 1)

#Regression using RF
rf = RandomForestRegressor(n_estimators = 100, random_state = 1)
rf.fit(xr, y)


Gini_pred=rf.predict(xr) #Predicted values of Gini Index
rf.score(xr,y)			#Score


# Get root mean square error of the prediction
rmse_rf = np.sqrt(mean_squared_error(y, Gini_pred))


#--Figure--
figt1=plt.scatter(x,y,color=colors[0]) #black t1
figPredt1=plt.scatter(x,Gini_pred,color=colors[2]) #Predicted Value
plt.legend((figt1, figPredt1),('Oridinal data', 'Predicted data')),
plt.xlabel('Unemployment(% of total labor force)'); plt.ylabel('GINI Index');
#plt.grid()
plt.show()



#-----Average Gini of Predicted data ------------

#--Get Avg of Gini index in Period t1
tst=dfp["GINI Index"].iloc[43:56]

#i=0
avg_Gpred=[]
for i in range(len(tst)):             #len(tst[0]) 6
	k= np.nanmean(tst.iloc[:,i].values)   #k= np.mean(tst.iloc[:,i].values) 
	avg_Gpred.append(k)


import pygal  #-> A package used to make the world map (vectorial images etc)

# create a world map
worldmap =  pygal.maps.world.World()
  
# set the title of the map
worldmap.title = 'A Duran -  Avg. Gini Index RF - Europe [2009-2015]'
  

# adding the countries
worldmap.add('Avg. Gini Index' , {
        'be' : avg_Gpred[0],
        'dk' : avg_Gpred[1],
        'fi' : avg_Gpred[2],
        'fr' : avg_Gpred[3],
        'de' : avg_Gpred[4],
        'it' : avg_Gpred[5], 
        'nl' : avg_Gpred[6], 
        'no' : avg_Gpred[7],        
		'pt' : avg_Gpred[8],
		'es' : avg_Gpred[9],
		'se' : avg_Gpred[10],
		'ch' : avg_Gpred[11],
		'gb' : avg_Gpred[12]
})



# save into the file
worldmap.render_to_file('Gini_WesternEurope_t2.svg')  #This can be opened with Chrome






# --------EXTRA
# Check distribution of Unemployment data (t1 and t2)
# sns.distplot(df['Unemployment(% of total labor force)'].dropna())
# 
# It can be a bimodal or Gauss.  For the moment I'll assume Gauss and generate synth data from the assumption
# from random import gauss
# seed(1)
# Create random Gaussian values
# xf=[]
# for i in range(len(xr)):
# 	value = gauss(15, 5)  # mean=15  std=5
# 	xf.append(value)
# 	
# 
# plt.hist(xf) #Histogram new distribution
# 	
# xf=np.transpose(xf)
# xf=xf.reshape(-1, 1)
# 
# rf.fit(xf, y)
# Gini_predf=rf.predict(xf)
# rf.score(xf,y)
# 
# figt1=plt.scatter(x,y,color=colors[0]) #black t1
# figPredt1=plt.scatter(xf,Gini_predf,color=colors[2]) #Predicted Value
# plt.show()

