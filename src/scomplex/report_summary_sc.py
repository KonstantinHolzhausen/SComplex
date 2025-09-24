import numpy as np 
import pandas as pd 
import gudhi as gd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from collections import defaultdict

def create_persistence_information(simplex_tree):
    ordered_death_persistence_pairs=[]
    persistence_pairs=[]
    persistence_information=[]
    for persistence_pair in simplex_tree.persistence():
        ordered_death_persistence_pairs.append(persistence_pair)
    ordered_death_persistence_pairs.sort(key=lambda x: x[1][1] if x[1][1] != float('inf') else float('inf'), reverse=False)

    for i in simplex_tree.persistence_pairs():
        persistence_pairs.append(i)
    #make a tuple structure with each element of ordered_death_persistence_pairs and persistence_pairs
    persistence_information = list(zip(ordered_death_persistence_pairs, persistence_pairs))
    return persistence_information

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

def merge_vertex(simplex_death,updated_vertex_dictionary):
    vertex_coordinate1=updated_vertex_dictionary[simplex_death[0]]
    vertex_coordinate2=updated_vertex_dictionary[simplex_death[1]]
    updated_coordinate= ((vertex_coordinate1[0] + vertex_coordinate2[0])/2,  (vertex_coordinate1[1] + vertex_coordinate2[1])/2) #midpoint of all points in simplex_death
    for node in simplex_death:
        updated_vertex_dictionary[node] = updated_coordinate
    
    return updated_vertex_dictionary

def get_simplex_coordinates(simplex_tree,distance_matrix,cluster_size=2): #distance matrix will be used in future, kept in function as a reminer
    
    #Create clustering based on co_occurrence matrix(returns matrix and vertices)
    coordinate_dictionary = {}
    co_occurrence, vertices = create_cooccur_matrix(simplex_tree)
    clustering = AgglomerativeClustering(
                n_clusters=cluster_size, 
                metric="euclidean",
                linkage='ward'
            )
    labels = clustering.fit_predict(co_occurrence)
    num_clusters = len(set(labels))
    cluster_centers = [
        (3 * np.cos(2 * np.pi * i / num_clusters), 3 * np.sin(2 * np.pi * i / num_clusters)) 
        for i in range(num_clusters)
    ]
    cluster_members = defaultdict(list)
    for v, label in zip(vertices, labels):
        cluster_members[label].append(v)
    colors = plt.cm.Set3(np.linspace(0, 1, max(num_clusters, 3)))
    vertex_colors = {}
    for label, members in cluster_members.items():
        cx, cy = cluster_centers[label]
        cluster_color = colors[label % len(colors)]  # Get color for this cluster
        if len(members) == 1:
            coordinate_dictionary.update({members[0]:(cx, cy)})
            vertex_colors[members[0]] = cluster_color  # Assign color to single member
        else:
            radius = 1.0
            angle_step = 2 * np.pi / len(members)
            for i, v in enumerate(members):
                angle = i * angle_step
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                coordinate_dictionary.update({v:(x,y)}) 
            for v in members:
                vertex_colors[v] = cluster_color 
    #create (x,y) positions for every vertex. These positions are made as cocentric circles. Position based on the distance matrix
    #Once all cooredinates are made create a dictionary where the vertices are 
    return coordinate_dictionary, vertex_colors

def visualize_clustered_simplicial_complex(simplex_tree, persistence_information, variable_names, distance_matrix, filtration_threshold=0.01, 
                                         clusters=3, show_labels=True):
    """
    Visualize a clustered simplicial complex with filtration-based rendering.
    
    Parameters:
    - simplex_filtration: list of (simplex, filtration_value) tuples from simplex_tree.get_filtration()
    - variable_names: list of variable names
    - filtration_threshold: threshold for showing solid vs dotted elements
    - min_cluster_size: minimum size for a cluster
    - show_labels: whether to show variable labels
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect('equal')
    ax.axis('off')
    original_coordinate_dictionary,vertex_colors=get_simplex_coordinates(simplex_tree,distance_matrix,clusters)
    updated_vertex_dictionary=original_coordinate_dictionary.copy()
    triangles_within_threshold=0
    edges_within_threshold = 0
    for feature in persistence_information:
        dimension=feature[0][0]
        birth=feature[0][1][0]
        death=feature[0][1][1]
        if dimension==0: #dimension 0, vertex
            simplex_birth=feature[1][0][0]
            simplex_death=feature[1][1]
            if filtration_threshold > death:
                updated_vertex_dictionary=merge_vertex(simplex_death,updated_vertex_dictionary)   
        if dimension ==1: #edges
            if (filtration_threshold < death) and (filtration_threshold >= birth):
                simplex_birth=feature[1][0]
                x_points=[updated_vertex_dictionary[simplex_birth[0]][0],updated_vertex_dictionary[simplex_birth[1]][0]]
                y_points=[updated_vertex_dictionary[simplex_birth[0]][1],updated_vertex_dictionary[simplex_birth[1]][1]]
                ax.plot(x_points,y_points,color="blue",linewidth=2)
                edges_within_threshold += 1
        if dimension==2: #triangles
            if (filtration_threshold < death) and (filtration_threshold > birth):
                simplex_birth=feature[1][0]
                triangle_points = np.array([updated_vertex_dictionary[node] for node in simplex_birth])
                triangles_within_threshold += 1
                
                # Create triangle polygon
                triangle = plt.Polygon(triangle_points, color="lightgreen", edgecolor="black", alpha=0.5)
                ax.add_patch(triangle)
            
    # Step 10: Draw vertices with cluster colors
    for v, (x, y) in updated_vertex_dictionary.items():
        ax.scatter(x, y, s=400, c=[vertex_colors[v]], edgecolors='black', 
                  linewidth=2, zorder=10)
        
        if show_labels:
            position_to_nodes = defaultdict(list)
            for v, (x, y) in updated_vertex_dictionary.items():
                # Round coordinates to handle floating point precision
                pos_key = (round(x, 6), round(y, 6))
                position_to_nodes[pos_key].append(v)
            for pos_key, vertices in position_to_nodes.items():
                x, y = pos_key
                if len(vertices) == 1:
                    # Single vertex - use normal styling
                    v = vertices[0]
                    label = variable_names[v] if v < len(variable_names) else str(v)
                    ax.text(x, y + 0.1, label, ha='center', va='bottom', 
                        fontsize=9, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                else:
                    # Multiple vertices at same position - stack labels vertically
                    labels = []
                    for v in sorted(vertices):  # Sort for consistent ordering
                        label = variable_names[v] if v < len(variable_names) else str(v)
                        labels.append(label)
                    
                    stacked_label = '\n'.join(labels)
                    ax.text(x, y + 0.15, stacked_label, ha='center', va='bottom', 
                        fontsize=8, fontweight='bold', linespacing=1.2,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', 
                                    alpha=0.9, edgecolor='darkblue', linewidth=1))

    # Step 11: Add legend and statistics
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=vertex_colors[i % len(vertex_colors)], 
                   markersize=12, label=f'Cluster {i} ')
        for i in range(clusters)
    ]
    
    # Add filtration legend
    legend_elements.extend([
        plt.Line2D([0], [0], color='blue', linewidth=2.5, label=f'Edges >= {filtration_threshold:.3f}'),
        plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=1, label="Filtration Threshold"),
        plt.Polygon([(0, 0)], color='skyblue', alpha=0.5, label='Triangles'),
        plt.Polygon([(0, 0)], color='lightgreen', alpha=0.4, label='Higher simplices')
    ])
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Add statistics
    stats_text = f"""Filtration Statistics:
    Edges: {edges_within_threshold}
    Triangles: {triangles_within_threshold}"""
    
    legend_height = len(legend_elements) * 0.04  # Estimate legend height
    stats_y_position = max(0.85 - legend_height, 0.02)  # Don't go below 2% of plot

    ax.text(0.02, stats_y_position, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    # Use suptitle instead of title for better spacing
    fig.suptitle(f"Clustered Simplicial Complex (Filtration ≤ {filtration_threshold})", 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Leave space for the suptitle
    plt.show()

def create_cooccur_matrix(simplex_tree):
    """
    Create co-occurrence matrix from a simplex tree.
    
    Parameters:
    - simplex_tree: GUDHI SimplexTree object
    
    Returns:
    - co_occurrence: numpy array representing co-occurrence matrix
    """
    # Convert generator to list to access simplex and filtration values
    simplex_filtration = list(simplex_tree.get_filtration())
    filtrations=[]
    # Extract vertices and build proper co-occurrence matrix
    vertices = sorted({v for simplex, filtration in simplex_filtration for v in simplex})
    
    vertex_index = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    co_occurrence = np.zeros((n, n))
    filtrations=[]
    for simplex, filtration in simplex_filtration:
        filtrations.append(filtration)
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                vi, vj = vertex_index[simplex[i]], vertex_index[simplex[j]]
                co_occurrence[vi][vj] += 1
                co_occurrence[vj][vi] += 1
    
    return co_occurrence, vertices