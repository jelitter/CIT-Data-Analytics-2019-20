# 📚 Unsupervised Learning `(12-11-2019)`

We need to cluster inputs into some classes, based on some similarities:

```
x1       C1 = { x3, x4 }
x2
x3       C2 = { x1, x2, x5 }
x4
x5
```

We compare data points with each other.

- First we need to **decide which algorithm** to use for Clustering.
- A **distance** or **similarity** function:

  ```
  D(x1, x2) = 1 or 0
  ```

  One popular set of functions is the _**Minkowski**_ family of distance functions:

  [Minkowski formula](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/minkdist.htm)

  `p = 1` : Manhattan distance / similarity

  `p = 2` : Euclidean distance / similarity

  Example:

  ```
   (income, age)
  x1 = (40, 60)
  x2 = (50, 60)
  x3 = (70, 80)

  Mahattan distance (p = 1):

  D(x1, x2) = ( |40 - 50|^1 + |60 - 60|^1 )^(1/1) = 10 + 0 = 10
  D(x1, x3) = ( |40 - 70|^1 + |60 - 80|^1 )^(1/1) = 30 + 20 = 50

  🔍 x1 and x2 are more similar since their distance is lower.
  ```

  For text data we use the `Cosine` function. We convert strings to vectors.

  [Cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity)

  `D(A, B) = (A * B) / ( |A| * |B| )`

---

## Types of **Clustering Algorithms**:

1. **Partitional** Clustering.

   ### **_K-Means Algorithm_**: Very popular and widely used.

   _Proposed by Macqueen, 1967_.

   Given `K`, Construct a **partition** of a database `D` of `m` objects into `K` clusters.

   **`1`**. Randomly choose `K` data points to be the initial cluster centres.

   **`2`**. Assign each data point to the closest cluster centre.

   **`3`**. Re-compute the cluster centres using the current cluster membership.

   **`4`**. If a **_convergence_** criteria is not met, go to `2`.

   Example for **K = 3**

   `C1` = 2,10

   `C2` = 5,8

   `C3` = 1,2

|                | c1=(2, 10) | c2=(5, 8) | c3=(1, 2) | cluster assignment |
| -------------- | :--------: | :-------: | :-------: | :----------------: |
| `A1 = (2, 10)` |     0      |     5     |     9     |       **c1**       |
| `A2 = (2, 5)`  |     5      |     6     |     4     |       **c3**       |
| `A3 = (8, 4)`  |     12     |     7     |     9     |       **c2**       |
| `A4 = (5, 8)`  |     5      |     0     |    10     |       **c2**       |
| `A5 = (7, 5)`  |     10     |     5     |     9     |       **c2**       |
| `A6 = (6, 4)`  |     10     |     5     |     7     |       **c2**       |
| `A7 = (1, 2)`  |     9      |    10     |     0     |       **c3**       |
| `A8 = (4, 9)`  |     3      |     2     |    10     |       **c2**       |

**`3`**. Recompute centres: **Iteration 1:**

|    c1    |   c2    |   c3    |
| :------: | :-----: | :-----: |
| A1(2,10) | A3(8,4) | A2(2,5) |
|          | A4(5,8) | A7(1,2) |
|          | A5(7,5) |         |
|          | A6(6,4) |         |
|          | A8(4,9) |         |

Calculate new centres:

- Centres for Cluster 1 **C1**: (2, 10)

- Centres for Cluster 2 **C2**: Mean of all points in cluster:
  `((8+5+7+6+4)/5, (4+8+5+4+9)/5) = (6, 6)`

- Centres for Cluster 3 **C3**: Mean of all points in cluster:
  `(1.5, 3.5)`

**`GO TO STEP 2 AGAIN :`**

| New centres -> | c1=(2, 10) | c2=(6, 6) | c3=(1.5, 3.5) | cluster assignment |
| -------------- | :--------: | :-------: | :-----------: | :----------------: |
| `A1 = (2, 10)` |     0      |     8     |       7       |       **c1**       |
| `A2 = (2, 5)`  |     5      |     5     |       2       |       **c3**       |
| `A3 = (8, 4)`  |     12     |     4     |       7       |       **c2**       |
| `A4 = (5, 8)`  |     5      |     3     |       8       |       **c2**       |
| `A5 = (7, 5)`  |     10     |     2     |       7       |       **c2**       |
| `A6 = (6, 4)`  |     10     |     2     |       5       |       **c2**       |
| `A7 = (1, 2)`  |     9      |     9     |       2       |       **c3**       |
| `A8 = (4, 9)`  |     3      |     5     |       8       |       **c1**       |

**`3`**. Recompute centres: **Iteration 2:**

|    c1    |   c2    |   c3    |
| :------: | :-----: | :-----: |
| A1(2,10) | A3(8,4) | A2(2,5) |
| A8(4,9)  | A4(5,8) | A7(1,2) |
|          | A5(5,8) |         |
|          | A6(5,8) |         |
|          |         |         |

When centres do not change, we have **Convergence** and we can finish the algorithm. Not the case yet, so

**`GO TO STEP 2 AGAIN :`**

Calculate new centres:

- Centres for Cluster 1 **C1**: `(3, 9.5)`

- Centres for Cluster 2 **C2**: `(6.5, 5.25)`

- Centres for Cluster 3 **C3**: `(1.5, 3.5)`

`...`

|    c1    |   c2    |   c3    |
| :------: | :-----: | :-----: |
| A1(2,10) | A3(8,4) | A2(2,5) |
| A4(5,8)  | A5(5,8) | A7(1,2) |
| A8(4,9)  | A6(5,8) |         |
|          |         |         |

**`3`**. Recompute centres: **Iteration 4:**

Calculate new centres:

- Centres for Cluster 1 **C1**: `(3.67, 9)`

- Centres for Cluster 2 **C2**: `(7, 4.3)`

- Centres for Cluster 3 **C3**: `(1.5, 3.5)`

If we re-calculate distances, we see this time that **membership will not change, so the algorithm has converged**.

---

2. **Hierachycal** Clustering. (_not covered_)

   Top-down tree-like partitioning

3) **Model based** Clustering. (_not covered_)

---

## **Quality** of Clustering:

- **Internal Evaluation:**

  - **Inter-cluster distance**: Goal is to Maximize distance.

    Distance between clusters' centers.

  - **Intra-cluster distance**: Goal is to Minimize distance.

    Minimize distance among points inside a cluster.

- **External Evaluation:**
  We use a matrix for classification and clustering

  ```
              Actual    Yes         No
  Predicted             (+)        (-)

    Yes (+)           True-Pos     False-Pos

    No  (-)           False-Neg    True-Neg

  ```

  **`Rand Index`** `= (TP + TN) / ( TP + FP + FN + TN )`

  (_Used for Clustering_)

  **`Precission`** `= TP / ( TP + FP )`

  **`Recall =`** `TP / ( TP + FN )`

  (_Precission and Recall are used for Classification Model_).

  **`f-measure`** `= 2 * (precision * recall) / ( precision + recall)`
