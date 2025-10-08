import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("bank.csv", sep=';')
print(df.info())

df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])

df3['y'] = df3['y'].map({'yes': 1, 'no': 0})

sns.heatmap(df3.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

y = df3['y']
X = df3.drop('y', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)
y_pred_log = log.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
