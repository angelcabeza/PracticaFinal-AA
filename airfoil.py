#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Angel Cabeza y Jose Luis Oviedo 
"""
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

np.random.seed(1)

#############################################################################
# Lectura y preparación de los datos

print ("Leyendo los datos...")
# Lectura de los datos del problema
data = np.loadtxt( 'datos/airfoil_self_noise.dat' )

print ("Lectura realizada con exito!")

input("\n--- Pulsar tecla para continuar ---\n")

# División del conjunto inicial en dos subconjuntos train y test
# con train el 80% de los datos y test con el 20%.

print ("Vamos a dividir el dataset en train (80%) y test (20%)")
train, test = train_test_split(data, test_size = 0.20, random_state = 123)

# Separación de las etiquetas con los datos.
etiquetas_train = train[:,len(train[0])-1]
etiquetas_test = test[:,len(test[0])-1]

train = np.delete(train, -1, axis=1)
test = np.delete(test, -1, axis=1)

print("División realizada correctamente")

input("\n--- Pulsar tecla para continuar ---\n")
#############################################################################
# PREPROCESADO DE DATOS

# Comprobación de que todas las características utilizan el mismo tipo de datos
print('Columna                   Tipo de dato')
print('--------------------------------------------')
for i in range(5):
    print(i, '                  ', type(train[0][i]))

input("\n--- Pulsar tecla para continuar ---\n")

print ("Correlacion de pearson para cada caracteristica")
# Analisis de la correlacion de Pearson
Pearson = np.corrcoef(train, rowvar = False)
sns.heatmap(Pearson, annot = True)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Miramos si hay valores perdidos
print("\n¿Existen valores perdidos?: ", end='')
print(pd.DataFrame(np.vstack([train, test])).isnull().values.any())

input("\n--- Pulsar tecla para continuar ---\n")

# Analisis de la varianza de las características
print("Analisis de la varianza de las características")
print(pd.DataFrame(data).describe().to_string())

input("\n--- Pulsar tecla para continuar ---\n")

# Datos estandarizados para aquellos modelos que lo necesiten
print ("Estandarizando los datos...")
scaler = StandardScaler()
scaler.fit(train)
standar_train = scaler.transform(train)
standar_test = scaler.transform(test)

print("Estandarizacion realizada")

input("\n--- Pulsar tecla para continuar ---\n")

#############################################################################

#############################################################################
# Grid de parámetros para los 4 algoritmos usados

print("Ajustando los modelos esto podria tardar unos instantes...")
score = 'neg_mean_squared_error'

parametrosSGDR = [{'alpha': [0, 0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],'penalty':['l1', 'l2'],'eta0' : [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
parametrosB = [{'n_estimators': [50, 100, 300, 500]}]
parametrosRF = [{'n_estimators':[100,250,500,750,1000],'max_features':['auto','sqrt','log2']}]
parametrosSVC = [{'C':np.logspace(-1,2,4),'gamma':[10,1,0.1,0.01]}]
paramKernel = [{'C':[100,10,1,0.1],'kernel':['rbf','poly']}]

columns_sgdr = ['mean_fit_time', 'param_alpha', 'param_penalty','param_eta0','mean_test_score',
               'std_test_score', 'rank_test_score']

columns_b = ['mean_fit_time', 'param_n_estimators','mean_test_score',
               'std_test_score', 'rank_test_score']

columns_rf = ['mean_fit_time', 'param_n_estimators', 'param_max_features','mean_test_score',
               'std_test_score', 'rank_test_score']

columns_svr = ['mean_fit_time', 'param_C', 'param_gamma','mean_test_score',
               'std_test_score', 'rank_test_score']

columns_svrKernel = ['mean_fit_time', 'param_kernel','mean_test_score',
               'std_test_score', 'rank_test_score']

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    SGDR = GridSearchCV(SGDRegressor(random_state=1), parametrosSGDR, scoring = score)
    SGDR.fit(standar_train,etiquetas_train)
    print('\nCross Validation para Stocastig Gradient Descent\n', pd.DataFrame(SGDR.cv_results_,columns=columns_sgdr).to_string())
    B = GridSearchCV(AdaBoostRegressor(random_state=1, loss = 'square'), parametrosB, scoring = score)
    B.fit(train,etiquetas_train)
    print('\nCross Validation para AdaBoost\n', pd.DataFrame(B.cv_results_,columns=columns_b).to_string())
    RF = GridSearchCV(RandomForestRegressor(random_state=1),parametrosRF,scoring=score)
    RF.fit(standar_train,etiquetas_train)
    print('\nCross Validation para Random Forest\n', pd.DataFrame(RF.cv_results_,columns=columns_rf).to_string())
    SVRKernel = GridSearchCV(SVR(cache_size=1000),paramKernel,score)
    SVRKernel.fit(standar_train,etiquetas_train)
    print("\nCross Validation para SVM (para ver que kernel es mejor)\n: ",pd.DataFrame(SVRKernel.cv_results_,columns=columns_svrKernel).to_string())
    SVR = GridSearchCV(SVR(cache_size=1000),parametrosSVC,score)
    SVR.fit(standar_train,etiquetas_train)
    print("\nCross Validation para SVM ajuste de parámetros C y gamma\n: ",pd.DataFrame(SVR.cv_results_,columns=columns_svr).to_string())

input("\n--- Pulsar tecla para continuar ---\n")

print("\nLos mejores parametros para Stocastig Gradient Descent son: ", SGDR.best_params_)
print("CV-MSE para Stocastig Gradient Descent: ", -SGDR.best_score_)

print("\nLos mejores parametros para AdaBoost son: ", B.best_params_)
print("CV-MSE para AdaBoost: ", -B.best_score_)

print("\nLos mejores parametros para Random Forest son: ", RF.best_params_)
print("CV-MSE para Random Forest: ", -RF.best_score_)

print("\nLos mejores parametros para SVM son: ", SVR.best_params_)
print("CV-MSE para SVC: ", -SVR.best_score_)

input("\n--- Pulsar tecla para continuar ---\n")

#############################################################################
# Calculando las cotas con la desigualdad de Hoeffding
print("Desigualdades de Hoeffding para cada algoritmo y error RMSE Y R2:\n")

SGDRtest_pre = SGDR.predict(standar_test)
etest_SGDR = mean_squared_error(etiquetas_test,SGDRtest_pre)

DH_SGDR = etest_SGDR + np.sqrt((1/(2*len(standar_test))) * np.log(2/0.05))
print("\nValor de error Etest para SGD: ", etest_SGDR)
print("Cota Eout desigualdad de Hoeffding para SGD: ", DH_SGDR)
print("Valor de error R2 para SGD: ", r2_score(etiquetas_test,SGDRtest_pre))
print("Valor de error RMSE para SGD: ", mean_squared_error(etiquetas_test,SGDRtest_pre, squared = False))

Btest_pre = B.predict(test)
etest_B = mean_squared_error(etiquetas_test,Btest_pre)

DH_B = etest_B + np.sqrt((1/(2*len(test))) * np.log(2/0.05))
print("\nValor de error Etest para AdaBoost: ", etest_B)
print("Cota Eout desigualdad de Hoeffding para AdaBoost: ", DH_B)
print("Valor de error R2 para B: ", r2_score(etiquetas_test,Btest_pre))
print("Valor de error RMSE para B: ", mean_squared_error(etiquetas_test,Btest_pre, squared = False))

RFtest_pre = RF.predict(standar_test)
etest_RF = mean_squared_error(etiquetas_test,RFtest_pre)

DH_RF = etest_RF + np.sqrt((1/(2*len(standar_test))) * np.log(2/0.05))
print("\nValor de error Etest para RF: ", etest_RF)
print("Cota Eout desigualdad de Hoeffding para RF: ", DH_RF)
print("Valor de error R2 para RF: ", r2_score(etiquetas_test,RFtest_pre))
print("Valor de error RMSE para RF: ", mean_squared_error(etiquetas_test,RFtest_pre, squared = False))

SVMtest_pre = SVR.predict(standar_test)
etest_SVM = mean_squared_error(etiquetas_test,SVMtest_pre)

DH_SVM = etest_SVM + np.sqrt((1/(2*len(standar_test))) * np.log(2/0.05))
print("\nValor de error Etest para SVM: ", etest_SVM)
print("Cota Eout desigualdad de Hoeffding para SVM: ", DH_SVM)
print("Valor de error R2 para SVM: ", r2_score(etiquetas_test,SVMtest_pre))
print("Valor de error RMSE para SVM: ", mean_squared_error(etiquetas_test,SVMtest_pre, squared = False))

input("\n--- Pulsar tecla para continuar ---\n")

#############################################################################
# Apartado 10, gráficas para la validación de resultados

print ("Demostracion grafica de la precision de nuestros modelos")

plt.scatter(etiquetas_test, SGDRtest_pre)
plt.plot(etiquetas_test, etiquetas_test,label="Perfect Predictor",color='r')
plt.legend()
plt.title("Perfect predictor vs Our Stocasting Gradient Descent Predictor")
plt.xlabel("True SPL")
plt.ylabel("Predicted SPL")

plt.show()

plt.scatter(etiquetas_test, Btest_pre)
plt.plot(etiquetas_test, etiquetas_test,label="Perfect Predictor",color='r')
plt.legend()
plt.title("Perfect predictor vs Our AdaBoost Predictor")
plt.xlabel("True SPL")
plt.ylabel("Predicted SPL")

plt.show()

plt.scatter(etiquetas_test, RFtest_pre)
plt.plot(etiquetas_test, etiquetas_test,label="Perfect Predictor",color='r')
plt.legend()
plt.title("Perfect predictor vs Our RandomForest Predictor")
plt.xlabel("True SPL")
plt.ylabel("Predicted SPL")

plt.show()

plt.scatter(etiquetas_test, SVMtest_pre)
plt.plot(etiquetas_test, etiquetas_test,label="Perfect Predictor",color='r')
plt.legend()
plt.title("Perfect predictor vs Our SVM Predictor")
plt.xlabel("True SPL")
plt.ylabel("Predicted SPL")

plt.show()

#############################################################################