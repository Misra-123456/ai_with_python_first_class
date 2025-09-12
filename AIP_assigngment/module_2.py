import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-5, 5, 200)
y1 = 2*x + 1
y2 = 2*x + 2
y3 = 2*x + 3
plt.plot(x, y1, 'r-', label="y = 2x + 1")
plt.plot(x, y2, 'b--', label="y = 2x + 2")
plt.plot(x, y3, 'g:', label="y = 2x + 3")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])
plt.scatter(x, y, marker='+', color='b')
plt.xlabel("x values")
plt.ylabel("y values")
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv(r"F:\weight-height.csv")
print(df.head())
length = df["Height"].values
weight = df["Weight"].values
length_cm = length * 2.54
weight_kg = weight * 0.453592
mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)
print("Average length:", mean_length)
print("Average weight:", mean_weight)
plt.hist(length_cm, bins=20, color="skyblue", edgecolor="black")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()


import numpy as np
A = np.array([[1, 2, 3],  [0, 1, 4],   [5, 6, 0]])
A_inv = np.linalg.inv(A)
print("Inverse of A:    \n  ", A_inv)
I1 = np.dot(A, A_inv)
I2 = np.dot(A_inv, A)
print("A * A_inv:\n\n", np.round(I1, 2))
print("A_inv * A:\n\n\n", np.round(I2, 2))
