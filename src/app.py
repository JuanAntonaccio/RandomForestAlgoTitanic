# Arrancando hacer el pipeline del metodo
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd
import seaborn as sns
import pickle

from sklearn.ensemble import  RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Comenzando el proceso del Pipeline")

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv') 
