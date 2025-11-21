# DM Models 2 - Homework 3: K-Means Clustering & Recommender Systems

## Overview

This repository contains implementations for two major data mining tasks:
1. **K-Means Clustering** with multiple distance metrics (Euclidean, Cosine, Jaccard)
2. **Recommender Systems** using Probabilistic Matrix Factorization and Collaborative Filtering

Both implementations are designed to answer specific questions outlined in the homework assignment regarding algorithmic analysis, performance comparison, and optimal parameter selection.

---

## Task 1: K-Means Clustering Analysis
### Implementation Details

**File:** `kmeans_experiment.ipynb`

The K-Means algorithm is implemented from scratch with three distance metrics:

#### 1. **Distance Metrics**

- **Euclidean Distance**: Standard L2 distance
  ```
  d(x,y) = sqrt(Σ(xi - yi)²)
  ```

- **Cosine Distance**: 1 - Cosine Similarity
  ```
  d(x,y) = 1 - (x·y)/(||x||·||y||)
  ```

- **Generalized Jaccard Distance**: 1 - Jaccard Similarity
  ```
  d(x,y) = 1 - Σmin(xi,yi)/Σmax(xi,yi)
  ```

#### 2. **Algorithm Flow**

1. **Initialization**: Randomly select K data points as initial centroids
2. **Assignment**: Assign each point to nearest centroid based on chosen distance metric
3. **Update**: Recalculate centroids as mean of assigned points
4. **Convergence Check**: Stop when:
   - Centroid positions don't change (shift < tolerance)
   - SSE (Sum of Squared Errors) increases
   - Maximum iterations (500) reached

#### 3. **Key Features**

- **SSE Calculation**: Measures clustering quality
  ```python
  SSE = Σ(distance(point, assigned_centroid)²)
  ```

- **Cluster Accuracy**: Uses majority voting
  - Each cluster labeled with most common true label
  - Accuracy = correct predictions / total points

- **Performance Metrics**:
  - SSE (lower is better)
  - Accuracy (higher is better)
  - Iterations to convergence
  - Runtime

### Answering Homework Questions

**Q1: SSE Comparison**
- The code runs K-Means with all three metrics
- Results saved in `kmeans_results_summary.csv`
- Typically: **Euclidean < Cosine < Jaccard** (lower SSE = better clustering)

**Q2: Accuracy Comparison**
- Majority voting labels each cluster
- Compares predicted vs true labels
- Results show which metric best captures true data structure

**Q3: Convergence Analysis**
- Tracks iterations and runtime for each metric
- Combined stopping criteria ensures fair comparison
- Results reveal computational efficiency differences

**Q4: Terminating Conditions**
- Code supports four stopping modes:
  - `'combined'`: All conditions (default)
  - `'centroid'`: Only centroid change
  - `'sse_increase'`: Only SSE increase
  - `'maxiter'`: Fixed iterations only
- Can modify `stop_on` parameter to test each condition

**Q5: Observations**

Based on typical results:
- Euclidean: Fast convergence, moderate accuracy
- Cosine: Better for high-dimensional data, good accuracy
- Jaccard: Slower, lower accuracy, sensitive to sparse data

---

## Task 2: Recommender Systems

### Implementation Details

**File:** `Recommender_Experiments.ipynb`

Uses the **Surprise** library to implement three recommender algorithms:

#### 1. **Algorithms Implemented**

**a) Probabilistic Matrix Factorization (PMF)**
- Approximated using `SVD` in Surprise
- Factors: 50 (user/item latent features)
- Learning rate: 0.005
- Regularization: 0.02
- Decomposes rating matrix: R ≈ U × V^T

**b) User-Based Collaborative Filtering**
- Finds similar users based on rating patterns
- Predicts ratings using weighted average of similar users' ratings
- Formula: `r̂(u,i) = r̄u + Σ(sim(u,v) × (rv,i - r̄v)) / Σ|sim(u,v)|`

**c) Item-Based Collaborative Filtering**
- Finds similar items based on user ratings
- Predicts ratings using weighted average of similar items
- Formula: `r̂(u,i) = Σ(sim(i,j) × ru,j) / Σ|sim(i,j)|`

#### 2. **Evaluation Metrics**

- **MAE (Mean Absolute Error)**:
  ```
  MAE = (1/n) × Σ|predicted - actual|
  ```
  Measures average prediction error magnitude

- **RMSE (Root Mean Square Error)**:
  ```
  RMSE = sqrt((1/n) × Σ(predicted - actual)²)
  ```
  Penalizes larger errors more heavily

#### 3. **Experimental Setup**

**Cross-Validation**: 5-fold CV
- Data split into 5 parts
- Each fold used once for testing, 4 for training
- Results averaged across folds

**Similarity Metrics Tested**:
- **Cosine**: Measures angle between rating vectors
- **MSD (Mean Squared Difference)**: Inverse of MSE
- **Pearson**: Correlation coefficient

**Neighbor Counts (K) Tested**: 5, 10, 20, 40, 80

### Answering Homework Questions

**Q2c: 5-Fold Cross-Validation Results**
```python
# Results structure (example):
# SVD (PMF):       RMSE=0.985, MAE=0.758
# User-KNN:        RMSE=0.968, MAE=0.745
# Item-KNN:        RMSE=0.935, MAE=0.721
```
Results saved in `results_similarity_table.csv`

**Q2d: Model Comparison**
- Best model determined by lowest RMSE/MAE
- Typically: **Item-KNN < User-KNN < SVD/PMF**
- Item-based often better because items have more consistent patterns

**Q2e: Similarity Metric Impact**
- Three metrics tested for both user/item-based
- Results plotted in `rmse_by_similarity.png`
- **Key Finding**: Impact varies between user/item-based
  - User-KNN: MSD usually best
  - Item-KNN: Also MSD, but differences may vary
- Cosine sensitive to rating scale

**Q2f: Number of Neighbors Impact**
- Tests K = [5, 10, 20, 40, 80]
- Results plotted in `rmse_vs_k.png`
- Shows performance curve for each metric

**Q2g: Optimal K Selection**
- Identified from `results_k_table.csv`
- Best K minimizes RMSE
- **Typical Results**:
  - User-KNN: K ≈ 10-20 (smaller neighborhoods better)
  - Item-KNN: K ≈ 40-80 (larger neighborhoods better)
- **Reason**: More items rated per user than users per item

---

## Installation & Requirements

### Required Libraries

```bash
pip install numpy==1.26.4
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-surprise
```

### Python Version
- Python 3.8+

---

## Running the Code

### Task 1: K-Means

```bash
jupyter notebook kmeans_experiment.ipynb
```

**Expected Outputs**:
1. Console output with SSE, accuracy, iterations, time for each metric
2. `kmeans_results_summary.csv` with comparison table

**Required Files**:
- `data.csv` (10,000 × 784 features)
- `label.csv` (10,000 labels)

### Task 2: Recommender Systems

```bash
jupyter notebook Recommender_Experiments.ipynb
```

**Expected Outputs**:
1. Console output with RMSE/MAE for each configuration
2. `results_similarity_table.csv` (similarity comparison)
3. `results_k_table.csv` (neighbor count experiments)
4. `rmse_by_similarity.png` (bar plot)
5. `rmse_vs_k.png` (line plot)

**Required Files**:
- `ratings_small.csv` (MovieLens dataset)

---

## Key Code Sections Explained

### K-Means: Distance Computation

```python
def euclidean_distances(X, C):
    # Vectorized computation: ||x-c||² = ||x||² - 2x·c + ||c||²
    X2 = np.sum(X**2, axis=1)[:, None]  # (n,1)
    C2 = np.sum(C**2, axis=1)[None, :]  # (1,k)
    cross = X @ C.T                      # (n,k)
    d2 = X2 - 2*cross + C2
    return np.sqrt(np.maximum(d2, 0))    # Avoid negative due to float errors
```

### K-Means: Convergence Check

```python
# Combined stopping criteria
if stop_on == 'combined':
    if max_shift <= self.tol:           # Centroids stable
        stop = True
    if prev_SSE and SSE > prev_SSE:     # SSE increased
        stop = True
```

### Recommender: Cross-Validation

```python
def cross_validate_mean(algo, data, measures=['rmse','mae'], cv=5):
    cv_results = cross_validate(algo, data, measures=measures, cv=cv)
    return np.mean(cv_results['test_rmse']), np.mean(cv_results['test_mae'])
```

### Recommender: Neighbor Sweep

```python
for k in [5, 10, 20, 40, 80]:
    sim_options = {'name': 'msd', 'user_based': True}
    algo = KNNBasic(k=k, sim_options=sim_options)
    rmse, mae = cross_validate_mean(algo, data)
    results.append({'k': k, 'rmse': rmse, 'mae': mae})
```

---

## Results Interpretation

### Task 1 Results

**Typical Performance Ranking**:
1. **SSE**: Euclidean < Cosine < Jaccard
2. **Accuracy**: Cosine ≈ Euclidean > Jaccard
3. **Convergence**: Euclidean (fastest) < Cosine < Jaccard (slowest)

**Insights**:
- Euclidean: Best for continuous, low-dimensional data
- Cosine: Better for high-dimensional, sparse data (like text)
- Jaccard: Good for binary/set data, struggles with continuous values

### Task 2 Results

**Typical Performance Ranking**:
1. **RMSE**: Item-KNN < User-KNN < PMF
2. **Best Similarity**: MSD performs best overall
3. **Optimal K**: Item-based needs larger K (40-80), User-based smaller (10-20)

**Insights**:
- Item-based CF leverages consistent item characteristics
- MSD works well because it's scale-invariant
- More neighbors help item-based due to rating sparsity

---
