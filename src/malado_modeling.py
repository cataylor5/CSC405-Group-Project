import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
project_root = Path(__file__).resolve().parents[1]
df = pd.read_csv(project_root / "data" / "trx-10k.csv")

# Clean status column
df["status"] = df["status"].str.lower()
df["status"] = df["status"].replace({
    "failed": "fail",
    "succeed": "success"
})

# Clean feature columns
df["city"] = df["city"].str.lower().str.strip()
df["card_type"] = df["card_type"].str.lower().str.strip()

df["city"] = df["city"].replace({
    "tehr@n": "tehran",
    "thr": "tehran",
    "thran": "tehran",
    "tehran ": "tehran"
})

df["card_type"] = df["card_type"].replace({
    "mastcard": "mastercard",
    "master card": "mastercard",
    "master-card": "mastercard",
    "vsa": "visa"
})

# Create hour feature
df["time"] = pd.to_datetime(df["time"])
df["hour"] = df["time"].dt.hour

# Correlation and p-values
df["status_encoded"] = df["status"].map({"fail": 0, "success": 1})

amount_corr, amount_p = pearsonr(df["amount"], df["status_encoded"])
hour_corr, hour_p = pearsonr(df["hour"], df["status_encoded"])

print("Amount correlation:", amount_corr)
print("Amount p-value:", amount_p)
print("Hour correlation:", hour_corr)
print("Hour p-value:", hour_p)

city_table = pd.crosstab(df["city"], df["status"])
_, city_p, _, _ = chi2_contingency(city_table)

card_table = pd.crosstab(df["card_type"], df["status"])
_, card_p, _, _ = chi2_contingency(card_table)

print("City chi-square p-value:", city_p)
print("Card type chi-square p-value:", card_p)

# Modeling setup
X = df[["card_type", "city", "amount", "hour"]]
y = df["status"]

X = pd.get_dummies(X, drop_first=True)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline model
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_acc = accuracy_score(y_test, tree_pred)

# Advanced model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Decision Tree Accuracy:", tree_acc)
print("Random Forest Accuracy:", rf_acc)

# Feature importance
importance = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importance.sort_values(ascending=False).head(10)

print("\nTop Features:")
print(top_features)