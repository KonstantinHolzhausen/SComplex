import numpy as np 
import pandas as pd 
import gudhi as gd
from hyppo.independence import Hsic
from scipy.stats import pearsonr
from dtw import *


def distance_correlation(x, y):
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

def compute_pairwise_metrics(df):
    results = []
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            x = df.iloc[:, i].values
            y = df.iloc[:, j].values

            pearson_val, _ = pearsonr(x, y)
            dcor_val = distance_correlation(x, y)
            try:
                dtw_val = dtw(x,y,distance_only=True)
            except ValueError:
                dtw_val= float("nan")
            try:
                hsic_val = Hsic().statistic(np.reshape(x,(-1,1)),np.reshape(y,(-1,1)))
            except ValueError:
                hsic_val=float("nan")

            results.append({
                "Var1": df.columns[i],
                "Var2": df.columns[j],
                "Pearson": pearson_val,
                "Distance_Correlation": dcor_val,
                "DTW" : dtw_val,
                "HSIC": hsic_val
            })

    return pd.DataFrame(results)
        #Note the dtw might need to be normalized here before compared to anything

def dtw_fix(results_df):
    results_df["DTWDistance"] = 1
    dtw_distance=[]

    for i in range(len(results_df)):
        try:
            #print(results_df["DTW"][i].normalizedDistance)
            dtw_distance.append(results_df["DTW"][i].normalizedDistance)
        except AttributeError:
            #print("index at failure is:" + str(i))
            dtw_distance.append(0)
    #dtw_distance
    results_df["DTWDistance"]=dtw_distance
    return results_df

def create_metric_matrix(results_df):
    variables = []
    seen = set()
    
    # First add variables from Var1 in order of appearance
    for var in results_df['Var1']:
        if var not in seen:
            variables.append(var)
            seen.add(var)
    
    # Then add any variables from Var2 that weren't in Var1
    for var in results_df['Var2']:
        if var not in seen:
            variables.append(var)
            seen.add(var)
    
    var_index = {var: i for i, var in enumerate(variables)}
    n = len(variables)
    # Step 2: Create distance matrices
    pearson_dist = np.ones((n, n))
    dcor_dist = np.ones((n, n))
    dtw_dist = np.ones((n, n))
    hsic_dist = np.ones((n, n))

    for _, row in results_df.iterrows():
        i, j = var_index[row['Var1']], var_index[row['Var2']]
        pearson_dist[i, j] = pearson_dist[j, i] =1- abs(row['Pearson'])
        dcor_dist[i, j] = dcor_dist[j, i] = 1-row['Distance_Correlation']
        dtw_dist[i, j] = dtw_dist[j, i] = 1 - row['DTWDistance']
        hsic_dist[i, j] = hsic_dist[j, i] = 1 - row['HSIC']
    np.fill_diagonal(pearson_dist, 0)
    np.fill_diagonal(dcor_dist, 0)
    np.fill_diagonal(dtw_dist, 0)
    np.fill_diagonal(hsic_dist, 0)
    return {"pearson_dist":pearson_dist,"dcor_dist":dcor_dist,"dtw_dist":dtw_dist,"hsic_dist":hsic_dist}


# Step 3: Build Vietoris–Rips complex from distance matrix
def build_rips_complex(dist_matrix, max_edge_length=3, max_dim=4):
    rips = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim)
    print(f"Number of simplices: {simplex_tree.num_simplices()}")
    diag = simplex_tree.persistence()
    gd.plot_persistence_diagram(diag)
    return simplex_tree, diag

def describe_persistence_features_table(stree, variables, min_lifetime=0.005, dimension=None):
    """
    Describe persistence features and return results as a pandas DataFrame.
    
    Parameters:
    - stree: GUDHI SimplexTree object
    - variables: list of variable names
    - min_lifetime: minimum lifetime threshold to include features
    - dimension: optional dimension filter
    
    Returns:
    - pandas DataFrame with persistence features
    """
    diagram = stree.persistence()
    pairs = stree.persistence_pairs()
    
    def name_simplex(simplex):
        if not simplex:
            return "∞ (persists forever)"
        named_nodes = [variables[i] for i in simplex]
        if len(simplex) == 1:
            return f"Node '{named_nodes[0]}'"
        elif len(simplex) == 2:
            return f"Edge between '{named_nodes[0]}' and '{named_nodes[1]}'"
        elif len(simplex) == 3:
            nodes_str = ", ".join(f"'{n}'" for n in named_nodes)
            return f"Triangle formed by {nodes_str}"
        else:
            nodes_str = ", ".join(f"'{n}'" for n in named_nodes)
            return f"{len(simplex)-1}D simplex with nodes {nodes_str}"
    
    # Collect feature data
    features_data = []
    
    for i, ((dim, lifetime), (birth_simplex, death_simplex)) in enumerate(zip(diagram, pairs)):
        birth_time = lifetime[0] if birth_simplex else float('inf')
        death_time = lifetime[1] if death_simplex else float('inf')
        feature_lifetime = death_time - birth_time
        
        # Apply filters
        if feature_lifetime >= min_lifetime:
            if dimension is None or dim == dimension:
                features_data.append({
                    'Feature_ID': i,
                    'Dimension': dim,
                    'Birth_Time': birth_time,
                    'Death_Time': death_time if death_time != float('inf') else '∞',
                    'Lifetime': feature_lifetime if feature_lifetime != float('inf') else '∞',
                    'Birth_From': name_simplex(birth_simplex),
                    'Death_From': name_simplex(death_simplex)
                })
    
    # Create DataFrame
    df_features = pd.DataFrame(features_data)
    
    # Sort by lifetime (descending) for most persistent features first
    if not df_features.empty:
        # Handle infinity values for sorting
        df_features['Lifetime_Sort'] = df_features['Lifetime'].apply(
            lambda x: float('inf') if x == '∞' else x
        )
        df_features = df_features.sort_values('Lifetime_Sort', ascending=False).drop('Lifetime_Sort', axis=1)
        df_features = df_features.reset_index(drop=True)
    
    return df_features