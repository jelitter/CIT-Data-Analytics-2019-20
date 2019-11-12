### numpy Matrixes 😎

To **create** a Matrix:

```Python
    np.matrix("1,2; 3,4")

    1 2
    3 4

    matrix1.size() # == 4
```

To **modify** a Matrix:

```Python
    np.matrix(matrixname, obj, index, axis)

```

Example:

```Python

  A = np.matrix("4, 5, 6; 2, 3, 4; 5, 6, 7")

  # 4 5 6
  # 2 3 4
  # 5 6 7


  col_new = np.matrix("2, 3, 4")

  # 2 3 4

  # To add the new column to the A matrix:

  np.insert(A, 0, col_new, axis = 0)

  # 2 4 5 6
  # 3 2 3 4
  # 4 5 6 7


  print(A[1, :])

  # 2 3 4

  print(B[2, 2])

  # 7


```
