import matplotlib.pyplot as plt
import numpy as np

# Generate an Array containing a 1000 random numbers from a
# Gaussian distribution
gaussian_numbers = np.random.normal(size=1000)
plt.hist(gaussian_numbers)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
