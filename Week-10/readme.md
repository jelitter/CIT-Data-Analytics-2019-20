# 📈 Visualization - `matplotlib` (19-11-2019)`

...

```py
#matplotlib notebook
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(17)
df = pd.DataFrame(np.random.randn(100,2))

df = df.cumsum()
df.plot()
plt.axis([-10, 120, -10, 20])
plt.show()
```

## Plot Command and Formatting

The default format string is 'b-' which is a solid blue line:

```py
import matplotlib.pyplot as plt

plt.plot([1,2,3,4], [1,4,9,16], float)
...

```

## Plotting Multiple line using Plot Command

```py
import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
 t = np.arrange(0., 5., 0.2)

 # red dashes, blue squares and green triangles
 plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
 plt.show()
```

## Shortcut Styles in Pandas

```py
plt.plot(x,y, linewidth=2.0, marker='*', linestyle='-', color='black')
```

```py
plot.plot([1,2,3,4], [1,4,9,16], linewidth=2.0,
marker = '*', linestyle='-', color='black')
```

## Using Text in Figures

```py
xlabel() # add an axis label to the X-axis
ylabel() # add an axis label to the Y-axis
title()
subtitle()
text() # adds a text in some location
```

## Legend

```py
# locations: 0 to ... (0 = 'best')
py.legend(['legend'],['normal'], loc='upper left')
```

## x-ticks

```py
# Replaces the x-label with our cusstom labels
labels = ['First Class', 'Second Class', 'Third Class']
plt.xticks([1,2,3], labels)

```

# Bar Graph

```py
plt.bar([1,2,3], males)
plt.show()
```

```py

labels = ['First Class', 'Second Class', 'Third Class']
bar_width = 0.35

index = np.arrange(1,4) # [1,2,3]
plt.bar(index, males, bar_width, color = 'g')

plt.bar(index + bar_width, females, bar_width, color = 'b')

plt.xlabel('PClass')
plt.ylabel('Number of Deaths')
# ...

```

## Bar Graphs with Series Object

```py
np.random.seed(12)

# Option 1
s = pd.Series(np.random.rand(10))
s.plot(kind='bar')

# Option 2
df2 = pd.DataFrame(np.random.rand(10, 4),
                   columns=['a', 'b', 'c', 'd'])
df2.plot(kind='bar')

df2.plot(kind='bar', stacked=True)  # Optional stacking of a,b,c,d

```

# Scatter Plots

```py
N = 100

x = np.random.rand(N)
y = np.random.rand(N)
area = np.pi * (np.random.rand()*10)

newX = np.random.rand(N)
newY = np.random.rand(N)
newArea = np.pi * (np.random.rand()*10)

plt.scatter(x,y, s=area, color = 'r')
plt.scatter(newX, newY, s=newArea, color = 'g')

plt.show()
```

## Scatter Plots with DataFrames

# Histograms

```py
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
```

# Boxplots

# lmplots

The lmplot function in seaborn are used to visualize a linear relationship as
determined through linear regression.

# Correlation Matrix
