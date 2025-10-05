from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

results = {}
for features in [["bmi","s5"], ["bmi","s5","bp"], list(X.columns)]:
    X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.2, random_state=5)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[str(features)] = r2
    print(features, "MSE=%.2f" % mse, "R2=%.3f" % r2)

plt.bar(results.keys(), results.values(), color=['blue','green','red'])
plt.ylabel("R² Score")
plt.title("Model Performance Comparison")
plt.xticks(rotation=20)
plt.show()






import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("50_Startups.csv")

print("Columns:", df.columns.tolist())
print("\nCorrelation:\n", df.corr(numeric_only=True))

X = df[["R&D Spend", "Marketing Spend"]]
y = df["Profit"]

plt.scatter(df["R&D Spend"],df["Profit"], color="blue")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("R&D Spend vs Profit")
plt.show()

plt.scatter(df["Marketing Spend"], df["Profit"], color="green")
plt.xlabel("Marketing Spend")
plt.ylabel("Profit")
plt.title("Marketing Spend vs Profit")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\nTrain RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("Train R²:", r2_score(y_train, y_train_pred))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("Test R²:", r2_score(y_test, y_test_pred))










import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

data = pd.read_csv("Auto.csv")
X = data.drop(columns=["mpg", "name", "origin"]).fillna(0)
y = data["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = np.logspace(-3, 3, 20)

ridge_models = [Ridge(alpha=a).fit(X_train, y_train) for a in alphas]
ridge_r2_scores = np.array([r2_score(y_test, m.predict(X_test)) for m in ridge_models])
best_ridge_alpha = alphas[np.argmax(ridge_r2_scores)]

lasso_models = [Lasso(alpha=a, max_iter=10000).fit(X_train, y_train) for a in alphas]
lasso_r2_scores = np.array([r2_score(y_test, m.predict(X_test)) for m in lasso_models])
best_lasso_alpha = alphas[np.argmax(lasso_r2_scores)]

print(f"Best Ridge alpha: {best_ridge_alpha:.4f}, R²: {ridge_r2_scores.max():.3f}")
print(f"Best Lasso alpha: {best_lasso_alpha:.4f}, R²: {lasso_r2_scores.max():.3f}")

plt.figure(figsize=(8,5))
plt.plot(alphas, ridge_r2_scores, marker='o', label="Ridge R²")
plt.plot(alphas, lasso_r2_scores, marker='s', label="Lasso R²")
plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("R² score on test set")
plt.title("R² vs Alpha for Ridge and Lasso")
plt.legend()
plt.show()
