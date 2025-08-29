# SComplex - Simplicial Complex Analysis

A Python package for analyzing time series data using topological data analysis and simplicial complexes.

## Installation

### Installing UV Package Manager

UV is a fast Python package installer and resolver. Install it using one of the following methods:

**On Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative - using pip:**
```bash
pip install uv
```

### Setting up the Environment

1. **Clone the repository:**
```bash
git clone <repository-url>
cd SComplex
```

2. **Create and activate the UV environment:**
```bash
# Create virtual environment
uv venv

# Activate the environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

3. **Install dependencies:**
```bash
uv pip install -r requirements.txt
```

4. **Install the package in development mode:**
```bash
uv pip install -e .
```

## Source Code Structure (`src/`)

The `src/` directory contains the core modules of the SComplex package:

### `complex_processing/`
- **Purpose**: Core functionality for simplicial complex construction and analysis
- **Key Functions**: 
  - Vietoris-Rips complex construction
  - Persistence computation and analysis
  - Topological feature extraction
  - Distance matrix processing

### `report_summary_sc/`
- **Purpose**: Generates comprehensive reports and summaries of simplicial complex analysis
- **Key Functions**:
  - Statistical summaries of topological features
  - Persistence diagram analysis
  - Report generation and formatting
  - Result visualization and documentation

## Jupyter Notebook: SimplicialComplexTemplate.ipynb

### Overview
This notebook provides a complete workflow for analyzing time series data using topological data analysis. It demonstrates how to transform correlation matrices into simplicial complexes and visualize their topological features.

### Notebook Sections

#### 1. Data Loading and Preprocessing
- Loads multiple CSV files from the `data/` directory
- Merges time series data on timestamp columns
- Filters out low-variance and constant columns
- Applies standardization using `StandardScaler`

#### 2. Metric Computation
- Computes pairwise relationships between variables using:
  - **Pearson Correlation**: Linear relationships
  - **Distance Correlation**: Non-linear dependencies
  - **DTW Distance**: Temporal alignment similarities
  - **HSIC**: Independence testing

#### 3. Simplicial Complex Construction
- Converts correlation matrices to distance matrices
- Builds Vietoris-Rips complexes using GUDHI library
- Creates simplex trees for topological analysis

#### 4. Persistence Analysis
- Computes persistence diagrams showing feature lifetimes
- Identifies birth and death times of topological features
- Analyzes persistence pairs for different dimensions (0D, 1D, 2D)

#### 5. Visualization
- **Clustered Visualization**: Groups variables by co-occurrence patterns
- **Filtration-based Rendering**: Shows edges and triangles based on threshold
- **Merged Node Labels**: Handles variables that become topologically equivalent
- **Interactive Legends**: Color-coded clusters with statistics

### Key Features

#### Advanced Visualization
- **Node Clustering**: Uses agglomerative clustering on co-occurrence matrices
- **Dynamic Merging**: Merges nodes based on persistence analysis
- **Multi-level Labels**: Stacked labels for merged variables
- **Threshold Filtering**: Shows simplices within specified filtration values

#### Persistence-based Analysis
- **Feature Tracking**: Monitors birth/death of topological features
- **Dimensional Analysis**: Separate handling of vertices, edges, and triangles
- **Lifetime Filtering**: Focus on persistent features above threshold

### Output
- Interactive plots showing the topological structure of your data
- Persistence diagrams revealing important relationships
- Clustered visualizations highlighting variable groupings
- Statistical summaries of edges and higher-dimensional features

This template provides a complete framework for applying topological data analysis to your time series datasets, revealing hidden structural relationships that traditional correlation analysis might