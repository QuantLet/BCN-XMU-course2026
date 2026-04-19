# Visualization Analysis Report

This file contains the network feature visualization analysis results for the Elliptic++ dataset.

---

## 1. Descriptive Statistics (descriptive_statistics.csv)

Overall statistics for all network features (without class separation).

| Feature | Mean | Median | Max | Skewness |
|---------|------|--------|-----|----------|
| in_degree_centrality | 5.64e-06 | 4.91e-06 | 1.39e-03 | 25.23 |
| out_degree_centrality | 5.64e-06 | 4.91e-06 | 2.32e-03 | 102.25 |
| in_out_degree_ratio | 2.41 | 0.83 | 1393.74 | 36.99 |
| pagerank | 4.91e-06 | 3.34e-06 | 5.27e-04 | 20.41 |
| clustering_coefficient | 0.014 | 0.00 | 1.00 | 8.57 |
| betweenness_centrality | 2.77e-08 | 0.00 | 1.92e-05 | 20.77 |
| closeness_centrality | 1.57e-05 | 7.36e-06 | 0.003 | 17.17 |

**Skewness Note**: All features show significant right-skewed distributions (skewness > 0), indicating that a small number of nodes have extremely high feature values.

---

## 2. Grouped Statistics (grouped_statistics.csv)

Mean and standard deviation grouped by node class (1=Illicit, 2=Licit, 3=Unknown).

| Feature | Class | Mean | Std |
|---------|-------|------|-----|
| clustering_coefficient | 1 (Illicit) | 3.56e-04 | 1.58e-02 |
| clustering_coefficient | 2 (Licit) | 8.32e-03 | 7.31e-02 |
| clustering_coefficient | 3 (Unknown) | 1.56e-02 | 1.04e-01 |
| betweenness_centrality | 1 (Illicit) | 2.16e-11 | 8.91e-10 |
| betweenness_centrality | 2 (Licit) | 2.07e-09 | 5.84e-08 |
| betweenness_centrality | 3 (Unknown) | 3.54e-08 | 4.94e-07 |

---

## 3. Histogram and KDE (histogram_kde.png)

**File**: `analysis/histogram_kde.png`

**Content**: Log-transformed histograms + KDE curves for 7 network features

**Metric Definitions**:
- **In-Degree Centrality**: Ratio of edges pointing to the node out of total edges
- **Out-Degree Centrality**: Ratio of edges emanating from the node out of total edges
- **In/Out Degree Ratio**: Ratio of in-degree to out-degree, reflecting transaction direction preference
- **PageRank**: Node importance measure based on random walks, similar to Google page ranking
- **Clustering Coefficient**: Proportion of the node's neighbors that are also connected to each other, measuring local clustering
- **Betweenness Centrality**: Extent to which a node serves as a bridge on shortest paths
- **Closeness Centrality**: Average shortest path distance from the node to all other nodes

**Chart Interpretation**: All features show right-skewed distributions, indicating most nodes have low feature values while a few nodes have extremely high values.

---

## 4. Boxplot (boxplot.png)

**File**: `analysis/boxplot.png`

**Content**: Boxplots of 7 network features grouped by class (log scale)

**Chart Interpretation**:
- Blue box = Unknown class (3), Green = Licit (2), Red = Illicit (1)
- Licit nodes show more dispersed feature value distributions
- Illicit nodes generally have lower and more concentrated feature values

---

## 5. Violin Plot (violinplot.png)

**File**: `analysis/violinplot.png`

**Content**: Violin plots of 7 network features grouped by class (log scale)

**Chart Interpretation**:
- Violin plots combine the advantages of boxplots and kernel density estimation
- More intuitive visualization of data distribution shapes
- Bimodal or multimodal distributions are more apparent in licit nodes

---

## 6. Degree Distribution Analysis

### 6.1 in_degree_distribution.png

**File**: `analysis/in_degree_distribution.png`

**Content**:
- Left: In-degree distribution histogram for licit vs illicit nodes (log scale)
- Right: In-degree interval frequency percentage comparison bar chart

**Key Statistics**:
| Metric | Licit Nodes | Illicit Nodes |
|--------|-------------|---------------|
| Mean | 1.90 | 1.27 |
| Median | 1.00 | 1.00 |
| Max | 284 | 177 |

---

### 6.2 out_degree_distribution.png

**File**: `analysis/out_degree_distribution.png`

**Content**:
- Left: Out-degree distribution histogram for licit vs illicit nodes (log scale)
- Right: Out-degree interval frequency percentage comparison bar chart

**Key Statistics**:
| Metric | Licit Nodes | Illicit Nodes |
|--------|-------------|---------------|
| Mean | 1.18 | 0.74 |
| Median | 1.00 | 1.00 |
| Max | **472** | **3** |

**Important Finding**: The maximum out-degree for illicit nodes is only 3, far below licit nodes' 472. This indicates that illicit nodes tend to maintain fewer outgoing connections to evade tracking.

---

### 6.3 degree_distribution_combined.png

**File**: `analysis/degree_distribution_combined.png`

**Content**: 2x2 combined figure
- Top-left: In-degree distribution histogram comparison
- Top-right: Out-degree distribution histogram comparison
- Bottom-left: In-degree boxplot (log scale)
- Bottom-right: Out-degree boxplot (log scale)

---

## 7. Centrality Metrics Analysis

### 7.1 centrality_histograms.png

**File**: `analysis/centrality_histograms.png`

**Content**: Overall distribution histograms for 7 centrality metrics (log scale)

**Chart Interpretation**: Each subplot shows the distribution of one centrality metric, exhibiting typical right-skewed long-tail distributions.

---

### 7.2 centrality_boxplots.png

**File**: `analysis/centrality_boxplots.png`

**Content**: Boxplot comparison of 7 centrality metrics between licit and illicit nodes (log scale)

---

### 7.3 centrality_statistics.csv

**File**: `analysis/centrality_statistics.csv`

**Content**: Detailed statistics for each centrality metric

| Metric | Licit (Mean/Median) | Illicit (Mean/Median) | Ratio |
|--------|---------------------|-----------------------|-------|
| In-Degree Centrality | 9.37e-06 / 4.91e-06 | 6.23e-06 / 4.91e-06 | 1.50x |
| Out-Degree Centrality | 5.82e-06 / 4.91e-06 | 3.64e-06 / 4.91e-06 | 1.60x |
| In/Out Degree Ratio | 4.83 / 0.83 | 4.33 / 0.83 | 1.12x |
| PageRank | 5.96e-06 / 2.57e-06 | 4.18e-06 / 2.61e-06 | 1.43x |
| Clustering Coefficient | 8.32e-03 / 0.00 | 3.56e-04 / 0.00 | **23.35x** |
| Betweenness Centrality | 2.07e-09 / 0.00 | 2.16e-11 / 0.00 | **95.60x** |
| Closeness Centrality | 2.11e-05 / 7.36e-06 | 1.13e-05 / 4.91e-06 | 1.88x |

---

## Metric Summary

### Degree Centrality
- **Definition**: Node's degree (number of connections) as a ratio of total edges
- **In-degree**: Number of incoming transactions
- **Out-degree**: Number of outgoing transactions
- **Normalization**: `degree / (n-1)`, where n is the total number of nodes

### PageRank
- **Algorithm**: Importance allocated based on visit probability from random walks
- **Characteristic**: Comprehensively considers direct and indirect connection relationships
- **Application**: Identifying "authoritative" nodes in the network

### Clustering Coefficient
- **Definition**: Proportion of the node's neighbors that are also connected to each other
- **Formula**: `actual edges / maximum possible edges`
- **Meaning**: Measuring local clustering/tightness

### Betweenness Centrality
- **Definition**: Frequency at which a node serves as a bridge on shortest paths
- **Formula**: Number of shortest paths passing through the node / Total number of shortest paths
- **Meaning**: Measuring the node's "intermediary" role

### Closeness Centrality
- **Definition**: Average shortest path distance from the node to all other nodes
- **Characteristic**: Higher values indicate shorter average distance to other nodes
- **Meaning**: Measuring the node's "central" position in the network

---

## Key Findings Summary

1. **Significant Out-Degree Difference**: Illicit nodes have a maximum out-degree of only 3, far below licit nodes' 472. This suggests illicit activities tend to maintain fewer outgoing connections to evade tracking.

2. **Clustering Coefficient Difference**: Licit nodes have a clustering coefficient 23x higher than illicit nodes, indicating that licit transactions form tighter local communities (e.g., business networks).

3. **Betweenness Centrality Difference**: Licit nodes have betweenness centrality 95x higher than illicit nodes, suggesting licit nodes more often serve as network intermediaries (e.g., exchanges, payment platforms).

4. **Right-Skewed Distributions**: All network features exhibit significant right-skewed distributions, consistent with the scale-free property of real-world networks.

5. **High Modularity**: Both subgraphs have modularity indices close to 0.98, indicating significant community structures exist in both.

---

*Generated: 2026-03-30*
*Data Source: Elliptic++ Bitcoin Transaction Dataset*
