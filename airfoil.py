#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:03:34 2021

@author: Angel Cabeza y Jose Luis Oviedo 
"""
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

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

# Modelo Lineal
lineal = LinearRegression().fit(train, etiquetas_train)
Ecv_lineal = - cross_val_score(lineal, train, etiquetas_train, scoring = 'neg_mean_squared_error').mean()

# Ada-Boost
Boost = GradientBoostingRegressor(random_state=0)
Ecv_Boost = - cross_val_score(Boost, train, etiquetas_train, scoring = 'neg_mean_squared_error').mean()