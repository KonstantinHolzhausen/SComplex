
from data_processor import data_processor as dp
from metrics import metrics as met
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import networkx as nx

from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D
from hyppo.independence import Hsic
from matplotlib import cm
from dtw import *

import numpy as np 
import pandas as pd 
import os 
import gudhi as gd

folder_path = 'data/DpShipData'
data_processor=dp.DataProcessor(folder_path,"pickled_dfs")
data_processor.load_csvs("experiment0", default_load=False) #Just to test if works, pickle is already loaded anyway
df=data_processor.join_dataframes()
df=data_processor.clean_data(df) #fills Nans, normalizes and drops empty columns
print(df)
print(df.columns)
if data_processor.pickle_ == []:
    metrics=met.MetricCalculator(df)
    metrics.build_metrics_dataframe()

