#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:03:34 2021

@author: Angel Cabeza y Jose Luis Oviedo 
"""
import numpy as np
from sklearn.model_selection import train_test_split

# Lectura de los datos del problema
data = np.loadtxt( 'datos/airfoil_self_noise.dat' )

# División del conjunto inicial en dos subconjuntos train y test
# con train el 80% de los datos y test con el 20%.
train, test = train_test_split(data, test_size = 0.20, random_state = 123)