import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(11)
dfb = pd.DataFrame(np.random.randn(10, 5))
# generate the plot
dfb.boxplot()
