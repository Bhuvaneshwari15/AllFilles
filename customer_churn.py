import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
# Load your dataset
df = pd.read_csv('E:/customer_churn/bank.csv')
# Identify categorical columns (assuming they are of type 'object')
categorical_cols = df.select_dtypes(include=['object']).columns
# Apply Label Encoding to each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
# Assuming 'churn' is your target variable
X = df.drop('Churn', axis=1)
y = df['Churn']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
# Print the results
print("decision tree model\n")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)
# Visualize the Decision Tree
plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Not Churned', 'Churned'], rounded=True)
plt.show()


#pre pruned decision tree
model1 = DecisionTreeClassifier(max_depth=4, min_samples_split=50, random_state=42)
# Train the model
model1.fit(X_train, y_train)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred1)
conf_matrix = confusion_matrix(y_test, y_pred1)
classification_rep = classification_report(y_test, y_pred1)
# Print the results
print("pre pruned decision tree\n")
print(f"Accuracy of pre pruned decision tree: {accuracy:.2f}")
print("Confusion Matrix of pre pruned decision tree:\n", conf_matrix)
print("Classification Report of pre pruned decision tree:\n", classification_rep)
# Visualize the pre-pruned Decision Tree
plt.figure(figsize=(12,8))
plot_tree(model1, filled=True, feature_names=X.columns, class_names=['Not Churned', 'Churned'], rounded=True)
plt.show()


#post pruned tree
model2 = DecisionTreeClassifier(max_depth=4, min_samples_split=50, random_state=42)
# Train the model
model2.fit(X_train, y_train)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred2)
conf_matrix = confusion_matrix(y_test, y_pred2)
classification_rep = classification_report(y_test, y_pred2)
# Print the results
print("post pruned trees\n")
print(f"Accuracy of post pruned tree: {accuracy:.2f}")
print("Confusion Matrix of post pruned tree:\n", conf_matrix)
print("Classification Report of post pruned tree:\n", classification_rep)
# Visualize the pre-pruned Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model2, filled=True, feature_names=X.columns, class_names=['Not Churned', 'Churned'], rounded=True)
plt.show()
