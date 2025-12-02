import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from utils import load_dataset, evaluate_model

csv_path = 'data/kaggle_dataset/WineQT.csv'
target_column = 'quality'
n_splits = 5
pca = False
pca_components = 3
ignore_columns = ['Id']

folds = load_dataset(csv_path,target_column,n_splits,True,pca,pca_components,ignore_columns)

#Pick best K using
#TODO: continue from here

k_values = list(range(1,31))
scores = []

for k in k_values:
    pipe = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=k))
    score = cross_val_score()