# 🤖 Machine Learning - `sklearn` (12-03-2019)`

**Training data** -> high accuracy

**Test data** -> low accuracy

## Cross-validation

It's `k-fold` (3, 5, 10...):

- Divide training data into 3 parts. A popular strategy for this is called **Stratified cross validation**.

---

- **Libraries:** `scipy`, `numpy`, `matplotlib`, `pandas`, `sklearn`

---

## ⚠ **Step 16** in Step-By-Step-ML-Models.ipynb

**Split-out validation dataset**

    array = dataset.values
    X = array[:,0:4]
    Y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)
