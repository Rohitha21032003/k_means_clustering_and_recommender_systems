# DM Models 2 - Homework 3: K-Means Clustering & Recommender Systems

## Overview

This repository contains implementations for two major data mining tasks:
1. **K-Means Clustering** with multiple distance metrics (Euclidean, Cosine, Jaccard)
2. **Recommender Systems** using Probabilistic Matrix Factorization and Collaborative Filtering

Both implementations are designed to answer specific questions outlined in the homework assignment regarding algorithmic analysis, performance comparison, and optimal parameter selection.

---
## Repository Structure
```
.
├── README.md
├── kmeans_experiment.ipynb          # Task 1: K-Means implementation
├── Recommender_Experiments.ipynb    # Task 2: Recommender systems
├── data.csv                         # K-Means features (10,000 × 784)
├── label.csv                        # K-Means true labels
├── ratings_small.csv                # MovieLens ratings dataset
├── kmeans_results_summary.csv       # Output: K-Means metrics
├── results_similarity_table.csv     # Output: Similarity comparison
├── results_k_table.csv              # Output: K values experiment
├── rmse_by_similarity.png           # Plot: Similarity impact
└── rmse_vs_k.png                    # Plot: Neighbor count impact
```
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
- Results: **Euclidean: 25,414,767,689.96 | Cosine: 686.23 | Jaccard: 4,239.95**
- **IMPORTANT**: SSE values cannot be directly compared across different distance metrics
- Each metric operates on a different numerical scale:
  - Euclidean produces large values in high-dimensional spaces
  - Cosine outputs bounded values in [0,2]
  - Jaccard also produces smaller bounded values
- The magnitude of SSE does NOT indicate which metric is "better"

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

Based on experimental results:
- **Cosine: Best accuracy (0.6279)** - optimal for high-dimensional data
- **Euclidean: Moderate accuracy (0.5851)** - fast, stable convergence (~33 iterations)
- **Jaccard: Poor accuracy (0.4442)** - converges very quickly (2 iterations) but produces low-quality clusters
- **Key Insight**: Jaccard terminates early due to SSE fluctuation, not because it clusters better

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
  - **Both User-KNN and Item-KNN**: MSD performs best consistently
  - For Item-KNN: MSD (0.9352) < Pearson (0.9900) < Cosine (0.9946)
  - For User-KNN: MSD (0.9684) < Cosine (0.9934) < Pearson (0.9980)
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

**Performance Summary**:

| Metric | SSE | Accuracy | Iterations | Time |
|--------|-----|----------|------------|------|
| Euclidean | 25,414,767,689.96 | 0.5851 | 33 | 4.44s |
| Cosine | 686.23 | 0.6279 | 29 | 4.99s |
| Jaccard | 4,239.95 | 0.4442 | 2 | 2.12s |

**Key Findings**:

1. **SSE Values Cannot Be Directly Compared**:
   - Euclidean produces very large SSE values due to L2 distance scale in high-dimensional space
   - Cosine distance outputs values in [0,2], resulting in naturally smaller SSE
   - Jaccard also produces bounded range values
   - **Conclusion**: SSE magnitude does not indicate which metric is "better" - each minimizes its own objective on different scales

2. **Accuracy Ranking**: **Cosine (0.6279) > Euclidean (0.5851) > Jaccard (0.4442)**
   - **Cosine is the best performer** for this high-dimensional dataset
   - Captures vector direction while ignoring magnitude - ideal for image/text-like features
   - Euclidean assumes spherical clusters, which may not fit the true data structure
   - Jaccard is inappropriate for continuous, dense feature vectors

3. **Convergence Behavior**:
   - **Euclidean & Cosine**: Stable convergence (~30 iterations, ~4-5 seconds)
   - **Jaccard**: Terminates extremely early (2 iterations, 2.12s) due to rapid SSE fluctuation, not superior performance
   - Cosine slightly slower per iteration due to vector normalization overhead

**Insights**:
- **Best Overall Method**: Cosine similarity for high-dimensional continuous data
- **Why Euclidean Underperforms**: Raw pixel space doesn't reflect true similarity structure
- **Why Jaccard Fails**: Designed for binary/set data, struggles with continuous numeric features
- **Theoretical Note**: Arithmetic mean centroids are mathematically optimal only for Euclidean distance; cosine/Jaccard would theoretically benefit from specialized centroid definitions or k-medoids

---

### Task 2 Results

**Model Performance Comparison (5-Fold Cross-Validation)**:

| Model | RMSE | MAE |
|-------|------|-----|
| PMF (SVD) | 0.9852 | 0.7577 |
| User-KNN (best) | 0.9684 | 0.7446 |
| Item-KNN (best) | 0.9352 | 0.7214 |

**Performance Ranking**: **Item-KNN < User-KNN < PMF** (lower is better)

**Similarity Metric Comparison**:

| Algorithm | Cosine | MSD | Pearson | Best |
|-----------|--------|-----|---------|------|
| User-KNN | 0.9934 | **0.9684** | 0.9980 | MSD |
| Item-KNN | 0.9946 | **0.9352** | 0.9900 | MSD |

- **MSD performs best** for both User-KNN and Item-KNN
- **Impact is consistent** across both algorithms
- Cosine and Pearson are competitive but slightly worse
- For Item-KNN: MSD (0.9352) < Pearson (0.9900) < Cosine (0.9946)

**Optimal Number of Neighbors (K)**:

**User-KNN**:
- Best K: **10-20** (k=10 gives RMSE=0.9634)
- Performance degrades with very small (k=5) or very large (k=80) neighborhoods

**Item-KNN**:
- Best K: **80** (RMSE=0.9315)
- Performance consistently improves as K increases
- Benefits from larger neighborhoods

| K | User-KNN (MSD) | Item-KNN (MSD) |
|---|----------------|----------------|
| 5 | 0.9853 | 1.0233 |
| 10 | **0.9634** | 0.9736 |
| 20 | 0.9628 | 0.9473 |
| 40 | 0.9670 | 0.9360 |
| 80 | 0.9747 | **0.9315** |

**Key Findings**:

1. **Best Overall Model**: **Item-Based Collaborative Filtering (MSD, k=80)**
   - RMSE: 0.9315, MAE: 0.7214
   - Outperforms both User-KNN and PMF

2. **Why Item-KNN Performs Best**:
   - Movies have more stable and consistent characteristic patterns
   - User preferences are more variable and context-dependent
   - Rating matrix structure: more ratings per user than users per item

3. **Why MSD Works Best**:
   - Scale-invariant: handles different rating scales naturally
   - Less sensitive to outliers compared to Pearson
   - Computationally simpler than cosine normalization

4. **Neighborhood Size Behavior**:
   - **User-KNN**: Prefers smaller neighborhoods (10-20)
     - Too many neighbors introduce noise from dissimilar users
   - **Item-KNN**: Benefits from larger neighborhoods (80+)
     - More items → more stable similarity patterns
     - Rating sparsity mitigated by broader item coverage

5. **Best K is NOT the Same**:
   - User-based: k ≈ 10-20
   - Item-based: k ≈ 80
   - Reflects fundamental differences in user vs. item similarity structure

**Insights**:
- **For MovieLens data**: Item-based CF >> User-based CF > PMF
- **Practical Recommendation**: Use Item-KNN with MSD similarity and k=80
- **PMF (SVD) Limitations**: May need more tuning (factors, regularization) to compete with CF methods
- **Similarity Metric Matters**: MSD's consistent superiority suggests rating scale normalization is critical

---
