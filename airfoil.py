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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Función para calcular el error de Cross Validation
def Cross_Validation(X, Y, iteraciones, modelo):
    tam = len(X)//iteraciones
    Error = 0
    
    for i in range(iteraciones):
        
        x_train = np.vstack((X[0:tam*i,:],X[tam*(i+1):,:]))
        x_test = X[tam*i:tam*(i+1),:]
        
        y_train = np.hstack((Y[0:tam*i],Y[tam*(i+1):]))
        y_test = Y[tam*i:tam*(i+1)]
        
        modelo.fit(x_train, y_train)
        
        prediccion = modelo.predict(x_test)
    
        Error = Error + mean_squared_error(prediccion, y_test)
        
    return Error/iteraciones

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

# Analisis de la varianza de las características
analisis = pd.read_csv('datos/airfoil_self_noise.dat', sep = '\s+', header = None)
pd.set_option('display.max_columns', 6)
print(analisis.describe())

# Datos estandarizados para aquellos modelos que lo necesiten
scaler = StandardScaler()
scaler.fit(train)
standar_train = scaler.transform(train)
standar_test = scaler.transform(test)

# Modelo Lineal
lineal = LinearRegression().fit(train, etiquetas_train)
Ecv_lineal = Cross_Validation(train, etiquetas_train, 5, lineal)