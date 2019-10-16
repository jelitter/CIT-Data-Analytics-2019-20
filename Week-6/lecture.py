import numpy as np

# arr = np.zeros((3, 3))
# print(arr)


# arr1 = np.arange(5, dtype=float)
# print(arr1)

# arr2 = np.arange(0, 102, dtype=float)  # 1D
# arr3 = arr2.reshape((10, 10))         # 2D
# print(arr2)
# print(arr3)


# # Use np.zeros to create an array with 12 rows and three columns. Print out the array.

# zeros = np.zeros((12, 3), int)
# print(zeros)

# zeros_reshaped = zeros.reshape(zeros, (6, 6))


arr1 = np.array([10, 20, 30], float)
arr2 = np.array([2, 3, 4], float)

print(arr1*arr2)

print(arr1 > arr2)  # [True True True]
print(arr1 == arr2)  # [False False False]

arr3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
even = arr3[:] % 2 == 0
print(arr3[even])  # [2, 4, 6, 8, 10]

# -------------------

data = np.array([[1, 2, 3], [2, 4, 5], [4, 5, 7], [6, 2, 3]], float)

resultA = data[:, 0] > 3
resultB = data[:, 2] > 6

print("\nDATA\n", data)
print("\n1\n", data[resultA])
print("\n2\n", data[resultB])
print("\n3\n", data[resultA & resultB])


# NOTES AFTER SLIDES
