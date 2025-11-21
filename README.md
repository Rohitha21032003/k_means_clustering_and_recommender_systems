# K-Means Clustering & Recommender Systems

This repository contains my solutions for **DM Models HW3**, divided into two major tasks:

- **Task 1:** Implementing and analyzing K-Means clustering completely from scratch  
- **Task 2:** Building a movie recommender system using Surprise (MovieLens dataset)

---

## Task 1 – K-Means Clustering (From Scratch)

This part implements a **full manual K-Means algorithm**, evaluates it using different similarity metrics, and analyzes convergence.

---

### Distance / Similarity Functions

Each metric is implemented as a separate function and plugged into K-Means:

```python
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def cosine_distance(x, y):
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def jaccard_distance(x, y):
    intersection = np.sum(np.minimum(x, y))
    union = np.sum(np.maximum(x, y))
    return 1 - intersection / union
```
### K-Means Algorithm Implementation

The entire pipeline is built manually:
```python
def kmeans(X, K, distance_fn, max_iters=500):
    centroids = initialize_centroids(X, K)
    
    for iteration in range(max_iters):
        labels = assign_clusters(X, centroids, distance_fn)
        new_centroids = update_centroids(X, labels, K)
        sse = compute_sse(X, labels, centroids, distance_fn)

        if np.allclose(centroids, new_centroids):
            break
        if iteration > 0 and sse > prev_sse:
            break
        
        prev_sse = sse
        centroids = new_centroids
    
    return labels, centroids, sse, iteration
```

### Experiments Performed

#### Q1 — SSE Comparison
- Run K-Means with:
  - Euclidean
  - Cosine
  - Jaccard
- Compare which achieves the lowest SSE.

#### Q2 — Accuracy Comparison
Clusters are labeled by majority vote:
```python
cluster_label = mode(true_labels[cluster_points])
```

Accuracy is computed vs. ground truth.

#### Q3 — Convergence Speed
Measured:
- Iterations
- Runtime
- Metric requiring most steps to converge

#### Q4 — Stopping Conditions
K-Means tested with:
- Centroid no-change
- SSE increases
- Max iterations

#### Q5 — Observations
Includes comparisons on:
- Stability
- Metric behavior
- Convergence patterns
- Accuracy vs. SSE

All code and outputs are available in:
kmeans_experiment.ipynb

---

## Task 2 – Recommender Systems (MovieLens)

This task uses Surprise to build User-based CF, Item-based CF, and PMF models using `ratings_small.csv`.

### Dataset Loading
```python
from surprise import Dataset, Reader

reader = Reader(line_format='user item rating timestamp', sep=',')
data = Dataset.load_from_file('ratings_small.csv', reader=reader)
```

### Models Implemented

#### 1. User-Based Collaborative Filtering
```python
algo = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
```

#### 2. Item-Based Collaborative Filtering
```python
algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
```

#### 3. Probabilistic Matrix Factorization (PMF)
(Surprise SVD implementation)
```python
from surprise import SVD
algo = SVD()
```

### Evaluation (MAE & RMSE)

All models use 5-fold cross-validation:
```python
from surprise.model_selection import cross_validate
results = cross_validate(algo, data, measures=['MAE', 'RMSE'], cv=5)
```

**Outputs:**
- Fold-wise MAE/RMSE
- Average MAE/RMSE

---

## Additional Experiments

### Similarity Metrics (Cosine, MSD, Pearson)

User-based & item-based CF evaluated with:
```python
for sim in ['cosine', 'msd', 'pearson']:
    algo = KNNBasic(sim_options={'name': sim, 'user_based': True})
```

**Plots show:**
- How similarity affects RMSE
- Whether item/user CF react similarly

### Effect of Number of Neighbors (K)

Evaluated for K = 5, 10, 20, 30, 40, 50:
```python
algo = KNNBasic(k=k, sim_options={...})
```

**Produces:**
- RMSE vs. K plot for User-CF
- RMSE vs. K plot for Item-CF

### Best K Selection

Identifies:
- Best K for User-based CF
- Best K for Item-based CF
- Whether both models share the same K

---

## Outputs

The notebooks generate:
- SSE tables
- Accuracy charts
- Iteration/runtime comparisons
- MAE/RMSE tables
- Similarity metric impact plots
- K-value performance plots

All summarized in my submission.

---
