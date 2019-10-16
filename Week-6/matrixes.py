import numpy as np


def title(text):
    print(f"\n\n⭐  {text}\n")


a = np.matrix("4, 5, 6; 2, 3, 4; 5, 6, 7")

title("A")
print(a)

# 4 5 6
# 2 3 4
# 5 6 7

col_new = np.matrix("2, 3, 4")
title("col_new")
print(col_new)

# 2 3 4

# To add the new column to the A matrix:

a = np.insert(a, 0, col_new, axis=1)

title("A after insert:")
print(a)


# 2 4 5 6
# 3 2 3 4
# 4 5 6 7

title("A[1, :]")
print(a[1, :])

# 2 3 4

title("A[2, 2]")
print(a[2, 2])

b = a[:3, :3]
title("B:")
print(b)

title("Matrix Determinant:")
print(np.linalg.det(b))


title("Matrix Inverse")
print(np.linalg.inv(b))

title("Matrix * Matrix Inverse")
print(np.matmul(b, np.linalg.inv(b)))

title("Matrix Transpose")
print(np.transpose(b))


# Equations

a = np.matrix(
    "0.24, 0.15, 0.18, 0.07; 0.65, 0.1, 0.24, 0.04; 0.1, 0.54, 0.42, 0.54; 0.01, 0.21, 0.18, 0.35")
b = np.matrix("75; 125; 200; 100")

title("A")
print(a)
title("B")
print(b)

title("Equations Solution")
print(np.linalg.solve(a, b))
