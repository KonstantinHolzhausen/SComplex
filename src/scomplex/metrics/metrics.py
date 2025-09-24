import pandas as pd
import numpy as np 
from scipy.stats import pearsonr
from hyppo.independence import Hsic
from dtw import *

from pathlib import Path
import iisignature

#########################################################################
#   Konstantins Code Block --  -- < START > -- 
#########################################################################

class DHSIC():
    """
    Functor retrieving the Hilbert Space Independence Criterion (HSIC) based distance metric.
    
    The HSIC is a distributional independence measure based on kernel mean embeddings of the distribution 
    in reproducing kernel Hilbert spaces (RKHS).

    The RKHS is defined by the kernel specified. If no kernel is specified, the default is the Gaussian 
    radial basis function (RBF) kernel.
    """

    @staticmethod
    def CenteredGramsFromKernel(kernel: callable, data: np.ndarray) -> np.ndarray:
        """
        Calculates the centered Gram matrix defined by the kernel evaluated on multichannel data.

        ARGS:
            kernel: callable (np.ndarray (#samples, #dimensions) , np.ndarray (#samples, #dimensions) -> np.ndarray (#samples, #samples), 
                normalised kernel function (values within [-1.0, 1.0]) yielding the gram matrix evaluated on finitely many samples
        """
        Nchannels, *rest = data.shape
        grams = []
        for c in range(Nchannels):
            gram = kernel(data[c], data[c]) # assumes that center
            gram = np.fill_diagonal(gram, 0.0) # centralise the normalised gram matrix
            grams.append(kernel(data[c], data[c]))
        return np.array(grams)

             

    @staticmethod
    def CenteredRBFGrams(data: np.ndarray) -> np.ndarray:
        """
        Computes the centered RBF kernel Gram matrices for multi channel data 
        that form the basis of an unbiased HSIC estimate (diagonal elements 0 instead of 1)

        Distances are based on the Euclidean metric. 
        Other metrics can be accounted for by adapting the squared distance calculation.

        ARGS:
            data: np.ndarray (#channels, #samples), data samples for all different channels
                            (#channels, #samples, #dims), data samples for all different channels and dimensions

        VALS:
            Ktilde: np.ndarray (#channels, #samples, #samples), modified
        """

        # calculate squared distances based on the dimensionality
        deg_tensor = len(data.shape) # degree of the data tensor
        if deg_tensor < 3:
            if deg_tensor == 2:
                Nc, _ = data.shape # Nc - number of channels, Nd - number of samples
                squared_dists = (data[:, None, :] - data[:, :, None])**2
            else:
                raise TypeError("data argument has to be of the format (#channels, #samples) or (#channels, #dims, #samples)")
        else:
            Nc, Nd, _ = data.shape #Nc - number of channels, Ndims - number of dimensions per sampels, #Nd - number of samples
            norms = np.einsum('cdi,cdi->cd', data, data) # norms tensor of size (#channels, #samples)
            dots = np.einsum('cmi,cni->cmn', data, data) # pairwise dot products between samples (#channels, #samples, #samples)
            squared_dists = norms[:, :, None] + norms[:, None, :]- 2.0*dots # squared euclidean distances

        # numerical safety
        np.maximum(squared_dists, 0.0, out=squared_dists)

        # calculate the median heuristic of the RBF kernel width
        sigmas = []
        for c in range(Nc):
            off_diag_dists = squared_dists[c][np.triu_indices_from(squared_dists[c])]
            # clean the data for non-zero distances to avoid biased estimates
            median = np.median(off_diag_dists[off_diag_dists > 0])
            sigmas.append(median if median > 0 else 1.0)

        sigmas = np.array(sigmas)
        sigmas = sigmas[:, None, None]
        K_tilde = np.exp(-squared_dists / sigmas)
        for c in range(Nc):
            np.fill_diagonal(K_tilde[c], 0.0)

        return K_tilde

    @staticmethod
    def unbiasedHSIC(data: np.ndarray, kernel: callable=None) -> np.ndarray:
        """
        Computes the unbiased estimator of the Hilbert Schmidt Independence Criterion 
        across multichannel data embedded into a Reproducing Kernel Hilbert Space 
        specified by the kernel function

        ARGS:
            data: np.ndarray (#channels, #samples), data samples for all different channels
                            (#channels, #dims, #samples)
            kernel: callable (np.ndarray (#channels, #samples) -> np.ndarray (#channels, #samples, #samples))

        VALS:
            HSIC matrix: np.ndarray (#channels, #channels)
        """

        if type(kernel) == type(None):
            kernel = DHSIC.CenteredRBFGrams

        deg_tensor = len(data.shape) # degree of the data tensor
        if deg_tensor < 3:
            if deg_tensor == 2:
                _, Nd = data.shape # Nc - number of channels, Nd - number of samples
            else:
                raise TypeError("data argument has to be of the format (#channels, #samples) or (#channels, #dims, #samples)")
        else:
            _, Nd, _ = data.shape #Nc - number of channels, Ndims - number of dimensions per sampels, #Nd - number of samples
        
        K_tilde = kernel(data)
        
        term1 = np.einsum("pij,qij->pq", K_tilde, K_tilde)
        
        S = np.sum(K_tilde, axis=(1, 2))
        term2 = S[:, None] @ S[None, :] / ((Nd-1)*(Nd-2))

        term3 = np.einsum("pij,qjl->piql", K_tilde, K_tilde)
        term3 = 2*np.sum(term3, axis=(1, 3)) / (Nd-2)

        return (term1 + term2 - term3) / (Nd*(Nd-1))

    @staticmethod
    def normalized_DHSIC(Hmatrix: np.ndarray, eps=1e-12, clip=True) -> np.ndarray:
        """
        Computes the normalised HSIC distance matrix from the unbiased HSIC estimator

        ARGS:
            Hmatrix: np.ndarray (#channels, #channels), unbiased HSIC estimator
            eps: float (1,), minimal offset to avoid divisions by zero
            clip: bool (1,), specifies whether clipping is whished or not


        VALS:
            DHSIC: np.ndarray (#channels, #channels), HSIC distance metric
        """

        hdiag = np.diag(Hmatrix).copy()
        hdiag = np.maximum(hdiag, 0.0) # avoid negatives on the diagonal
        scale = np.sqrt(np.outer(hdiag + eps, hdiag + eps)) # create normalisation matrix
        nHmatrix = Hmatrix / scale
        if clip:
            nHmatrix = np.clip(nHmatrix, 0.0, 1.0) # normalise to the interval
        return 1.0 - nHmatrix # distance from normalised HSIC metric matrix
    

    def __init__(self, kernel:callable=None):
        if kernel is None:
            self.getCenteredGrams = DHSIC.CenteredRBFGrams
        else:
            self.getCenteredGrams = lambda data: DHSIC.CenteredGramsFromKernel(kernel, data)

    def __call__(self, data: np.ndarray, normalize=True) -> np.ndarray:
        HSIC = DHSIC.unbiasedHSIC(data, kernel=self.getCenteredGrams)
        if normalize==True:
            HSIC = DHSIC.normalized_DHSIC(HSIC)
        return HSIC


#########################################################################
#   Konstantins Code Block --  -- < END > -- 
#########################################################################



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
