import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
df.to_numpy()
x = df[["Age", "Income"]]
y = df["Prediction"]

# Decision Tree
model = DecisionTreeClassifier()
model.fit(x.values, y.values)
x_new = [[30, 70000]]

predict = model.predict(x_new)
print("Decision Tree : "+predict)

# Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x.values, y.values)
predict1 = model.predict(x_new)
print("Random Forest : " +predict1)
