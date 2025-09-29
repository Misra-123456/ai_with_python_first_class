import numpy as np
import matplotlib.pyplot as plt
samples = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]
for size in samples:
    roll_a = np.random.randint(1, 7, size)
    roll_b = np.random.randint(1, 7, size)
    total = roll_a + roll_b
    counts, bins = np.histogram(total, bins=range(2, 14))
    plt.bar(bins[:-1], counts / size, width=0.8, color='green', edgecolor='black')
    plt.title(f"Dice Sum Distribution (Sample={size})")
    plt.xlabel("Sum of Two Dice")
    plt.ylabel("Relative Frequency")
    plt.show()
print("As sample size increases, the histogram smooths out and approaches the true probabilities of dice sums.")





import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
df = pd.read_csv(r"F:\weight-height.csv")
print(df.head())
heights = df["Height"].values.reshape(-1, 1)
weights = df["Weight"].values
regressor = LinearRegression()
regressor.fit(heights, weights)
predicted_weights = regressor.predict(heights)
plt.scatter(heights, weights, color="blue", alpha=0.4, label="Observed data")
plt.plot(heights, predicted_weights, color="orange", label="Fitted line")
plt.xlabel("Height (inches)")
plt.ylabel("Weight (lbs)")
plt.title("Linear Regression: Height vs Weight")
plt.legend()
plt.show()
rmse_val = np.sqrt(mean_squared_error(weights, predicted_weights))
r2_val = r2_score(weights, predicted_weights)
print("RMSE:", rmse_val)
print("R^2:", r2_val)
print("Generally, taller individuals weigh more. RMSE shows the typical prediction error in weight, while R^2 indicates how much of the weight variation is explained by height.")