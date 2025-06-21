# Iris Classification with Scikit-learn
# =====================================
# This script trains a Decision Tree classifier on the Iris dataset and evaluates its performance.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target  # Add target column

# Handle missing values (if any)
print("Checking for missing values...")
print(df.isnull().sum())  # No missing values in the Iris dataset
# df.dropna(inplace=True)  # Uncomment if cleaning is needed

# Encode categorical labels to numeric
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Split data into training and test sets
X = df.drop(columns=['species'])
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, average='macro'):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, average='macro'):.2f}")
