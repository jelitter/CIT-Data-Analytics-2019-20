# 🤖 Introduction to Machine Learning `(22-10-2019)`

> A computer program / machine is said to learn from experience with respect to **some class of task**, and a **performance meassure** `P` if (the learner's) performance at tasks in the class as measured by `P` , **improves with experience**. -- "Tom Mitchel, 1997"

- Also called **Inductive Learning**.

![img](http://giphygifs.s3.amazonaws.com/media/LAcLF2C7pyqPu/giphy.gif)

---

## Types of Machine Learning:

- ### **Supervised Learning**:

  ```
  x --> y = f(x)
  ```

  | Name       | Age |  Income | Gender | Class (Y/N) |
  | ---------- | :-: | ------: | :----: | :---------: |
  | Customer1  | 26  | \$41000 |   M    |      Y      |
  | Customer2  | 44  | \$52000 |   F    |      N      |
  | Customer3  | 32  | \$46000 |   M    |      Y      |
  | Customer4  | 33  | \$41000 |   F    |      N      |
  | ...        | ..  |     ... |  ...   |     ...     |
  | Customer15 | 33  | \$41000 |   F    |      N      |

  The goal is to find `f(x)` such at it divides `x` values in both classes (`Y`/`N`). We want the **_intercept point_** and the **_slope_**:

  ```
     Age  |   x                     /    x
          |                x       /
          |                       /   x
          |       [N]            /          [Y]
          |                     /    x
          |            x       /             x
          |     x             /         x
          |                  / x
          |_________________/___________________________
                           /                      Income

    Y: Bought a PC
    N: Didn't buy a PC

  ```

  - **Inductive Bias**: For every learning algorithm.
  - **Language Bias**: Will decide will `f` will be chosen (linear, curve, etc).
  - **Search Bias**:

--

- **_Classification_**: Label / Class is discrete (_e.g. Yes / No_).

* **_Regression_**: Label / Class is continuous (_e.g. Salary_).

* ### **Unsupervised Learning**:

  - **_Clustering_**: Grouping of similar items.

    Example: We have a 1M . records dataset, and want to take a sample of 1,000 items.
    _Clustering_ will create groups of data based on some characteristics (age, income, etc.).

  - **_Association_** or _Frequent Occurance_ [Rule Minning]

- **Reinforced Learning**: _(not covered)_

  ```
    x --[ ]--> y = f(x)

         ^--- Agent - use loss function to adjust y  (based on y'/y)
  ```

---

## Performance Measurement:

- **Classification**: `Error`
- **Regression**: `Error`
- **Clustering**: There is no standard metric

  One often used is `purity` (items in the same group should be similar)

- **Association**: `support` / `confidence`
