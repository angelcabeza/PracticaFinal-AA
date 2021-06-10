#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:03:34 2021

@author: Angel Cabeza y Jose Luis Oviedo 
"""
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


# Lectura de los datos del problema
data = np.loadtxt( 'datos/airfoil_self_noise.dat' )

# División del conjunto inicial en dos subconjuntos train y test
# con train el 80% de los datos y test con el 20%.
train, test = train_test_split(data, test_size = 0.20, random_state = 123)

# Separación de las etiquetas con los datos.
etiquetas_train = train[:,len(train[0])-1]
etiquetas_test = test[:,len(test[0])-1]

train = np.delete(train, -1, axis=1)
test = np.delete(test, -1, axis=1)

# Comprobación de que todas las características utilizan el mismo tipo de datos
print('Columna                   Tipo de dato')
print('--------------------------------------------')
for i in range(5):
    print(i, '                  ', type(train[0][i]))
   
# Analisis de la correlacion de Pearson
Pearson = np.corrcoef(train, rowvar = False)
sns.heatmap(Pearson, annot = True)
plt.show()

# Miramos si hay valores perdidos
print("\n¿Existen valores perdidos?: ", end='')
print(pd.DataFrame(np.vstack([train, test])).isnull().values.any())

# Analisis de la varianza de las características
print(pd.DataFrame(data).describe().to_string())

# Datos estandarizados para aquellos modelos que lo necesiten
scaler = StandardScaler()
scaler.fit(train)
standar_train = scaler.transform(train)
standar_test = scaler.transform(test)

# =============================================================================
# # Modelo Lineal
# lineal = LinearRegression().fit(train, etiquetas_train)
# Ecv_lineal = - cross_val_score(lineal, train, etiquetas_train, scoring = 'neg_mean_squared_error').mean()
# 
# # Ada-Boost
# Boost = GradientBoostingRegressor(random_state=0)
# Ecv_Boost = - cross_val_score(Boost, train, etiquetas_train, scoring = 'neg_mean_squared_error').mean()
# 
# =============================================================================
# Red neuronal
score = 'neg_mean_squared_error'

#parametrosMLP = [{'hidden_layer_sizes':[50,75,100],'alpha':np.logspace(-3,1,4),'max_iter':[10000,100000,1000000]}]
parametrosRF = [{'n_estimators':[10,100,250,500],'max_features':['auto','sqrt','log2']}]

# =============================================================================
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     MLP = GridSearchCV(MLPRegressor(activation='logistic'),parametrosMLP,scoring=score)
#     MLP.fit(train,etiquetas_train)
# =============================================================================

#print("Mejores parametros para MLP: ", MLP.best_params_)
#print('CV-MSE para Perceptron Multicapa: ', -np.mean(cross_val_score(MLP,train,etiquetas_train,scoring=score)))

columns_rf = ['mean_fit_time', 'param_n_estimators', 'param_max_features','mean_test_score',
               'std_test_score', 'rank_test_score']

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    RF = GridSearchCV(RandomForestRegressor(random_state=1),parametrosRF,scoring=score)
    RF.fit(train,etiquetas_train)
    print('Cross Validation para Regresión Logística\n', pd.DataFrame(RF.cv_results_,columns=columns_rf).to_string())


print("Los mejores parametros para Random Forest son: ", RF.best_params_)
print("CV-MSE para Random Forest: ", -RF.best_score_)

# GradientBoost
"""
param_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],  
              'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
              'criterion':['mse'],
              'max_depth':[1,2,3,4,5,6,7,8,9,10]}

grid = GridSearchCV(GradientBoostingRegressor(), param_grid, scoring = 'neg_mean_squared_error') 
grid.fit(train, etiquetas_train)

print(grid.best_params_) 
"""
Boost = GradientBoostingRegressor(criterion = 'mse', max_depth = 5, n_estimators = 1000, learning_rate = 0.2)

Ecv_Boost = - cross_val_score(Boost, train, etiquetas_train, scoring = 'neg_mean_squared_error').mean()
