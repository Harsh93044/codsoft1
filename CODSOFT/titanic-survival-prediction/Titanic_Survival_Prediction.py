# Step 1: Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 2: Load the data
df = pd.read_csv("Titanic-Dataset.csv")
print("First 5 rows:\n")
print(df.head())

# Step 3: Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Step 4: Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\nCleaned Data Info:\n")
print(df.info())

# Step 5: Encode categorical columns
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])            # male = 1, female = 0
df['Embarked'] = le.fit_transform(df['Embarked'])  # S = 2, C = 0, Q = 1

print("\nEncoded Data:\n")
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create logistic regression model
model = LogisticRegression(max_iter=1000)  # extra iterations to avoid warning
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
