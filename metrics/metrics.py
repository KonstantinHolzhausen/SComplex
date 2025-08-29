import pandas as pd
import numpy as np 
from scipy.stats import pearsonr
from hyppo.independence import Hsic
from dtw import *

class MetricCalculator:
    def __init__(self, dataframe: pd.DataFrame, verbose: bool = True):
        """
        Initialize with cleaned/normalized dataframe.
        """
        self.df = dataframe
        self.verbose = verbose
        self.metrics_df = pd.DataFrame()
        self.variables =self.df.columns

    def distance_correlation(self,x,y):
        """
        Compute distance correlation between all pairs of variables.
        """
        x = x[:, None]
        y = y[:, None]
        a = np.abs(x - x.T)
        b = np.abs(y - y.T)
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
        dcov = np.sqrt((A * B).mean())
        dvar_x = np.sqrt((A * A).mean())
        dvar_y = np.sqrt((B * B).mean())
        return dcov / np.sqrt(dvar_x * dvar_y) if dvar_x > 0 and dvar_y > 0 else 0

    def build_metrics_dataframe(self):
        """
        Combine all computed metrics into a single dataframe.
        Columns: var1, var2, correlation, distance_correlation, hsic, dtw
        """
        print("Computing metrics...")
        results=[]
        for i in range(len(self.df.columns)):
            for j in range(i + 1, len(self.df.columns)):
                x = self.df.iloc[:, i].values
                y = self.df.iloc[:, j].values
                pearson_val, _ = pearsonr(x, y)
                dcor_val = self.distance_correlation(x, y)
                try:
                    dtw_val = dtw(x,y,distance_only=True)
                except ValueError:
                    dtw_val= float("nan")
                try:
                    hsic_val = Hsic().statistic(np.reshape(x,(-1,1)),np.reshape(y,(-1,1)))
                except ValueError:
                    hsic_val=float("nan")

                results.append({
                    "Var1": self.df.columns[i],
                    "Var2": self.df.columns[j],
                    "Pearson": pearson_val,
                    "Distance_Correlation": dcor_val,
                    "DTW" : dtw_val,
                    "HSIC": hsic_val
                })
        results_df = pd.DataFrame(results)
        results_df["DTWDistance"] = 1
        dtw_distance=[]
        for i in range(len(results_df["DTW"][i].normalizedDistance)):
            try:
                dtw_distance.append(results_df["DTW"][i].normalizedDistance)
            except AttributeError:
                dtw_distance.append(0)
        results_df["DTWDistance"]=dtw_distance
        self.metrics_df=results
        print("Metrics computed: " +  str( self.metrics_df.columns[3:]))
    def get_metrics(self) -> pd.DataFrame:
        """
        Public method to run all metrics and return the final dataframe.
        """
        if self.verbose:
            self.build_metrics_dataframe()

        return self.metrics_df
