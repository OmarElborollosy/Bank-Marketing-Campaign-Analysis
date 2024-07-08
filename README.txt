# Bank Marketing Campaign Analysis

This project aims to analyze the Bank Marketing dataset and predict whether a customer will subscribe to a term deposit based on various demographic and socio-economic features. The analysis includes data preparation, exploratory data analysis (EDA), feature selection and engineering, building a decision tree classifier, hyperparameter tuning, evaluation, and interpretation of results.

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset used in this project is the Bank Marketing dataset from the UCI Machine Learning Repository. It contains information about a bank's direct marketing campaigns based on phone calls.

- [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The data includes the following features:
- `age`: Age of the customer
- `job`: Type of job
- `marital`: Marital status
- `education`: Level of education
- `y`: Whether the client has subscribed to a term deposit (binary: 'yes','no')

## Project Structure
The project consists of the following steps:

1. **Data Preparation**:
   - Load the dataset.
   - Check the first few rows, data types, and basic statistics.

2. **Exploratory Data Analysis (EDA)**:
   - Initial exploration of the dataset.

3. **Feature Selection and Engineering**:
   - Select relevant features and encode categorical variables.
   - One-hot encoding for `job`, `marital`, and `education`.

4. **Data Splitting**:
   - Split the dataset into training and testing sets.

5. **Model Building**:
   - Build a Decision Tree Classifier.
   - Evaluate the model using accuracy, precision, recall, and F1-score.

6. **Model Visualization**:
   - Visualize the decision tree.
   - Plot the confusion matrix.

7. **Hyperparameter Tuning **:
   - Tune hyperparameters using RandomizedSearchCV.
   - Train and evaluate the optimized model.

8. **Predictions**:
   - Make predictions on new data and visualize the results.

9. **Interpretation**:
   - Analyze feature importance and decision rules using SHAP values.
   - Export the decision tree.

## Installation
To run this project, you need to have Python installed along with the necessary libraries. You can install the required libraries using the following commands:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap graphviz
```

Ensure Graphviz is installed on your system. For installation instructions, visit the [Graphviz download page](https://graphviz.org/download/).

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/bank-marketing-analysis.git
   cd bank-marketing-analysis
   ```

2. **Run the analysis**:
   - You can run the provided script to perform the analysis step by step.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import shap
import graphviz

# Load and prepare the data
url = "https://archive.ics.uci.edu/dataset/222/bank+marketing"
file_path = "/kaggle/input/bankmarketing/bank/bank-full.csv"
df = pd.read_csv(file_path, sep=";")
df.head()
df.info()
df.describe()

# Feature selection and encoding
demographics_df = df[['age', 'job', 'marital', 'education', 'y']]
demographics_df['y'] = demographics_df['y'].map({'yes': 1, 'no': 0})
df_job_encoded = pd.get_dummies(demographics_df[['job']], drop_first=True)
df_marital_encoded = pd.get_dummies(demographics_df[['marital']], drop_first=True)
df_education_encoded = pd.get_dummies(demographics_df[['education']], drop_first=True)
demographics_df = demographics_df.drop(['job', 'marital', 'education'], axis=1)
X_encoded = pd.concat([demographics_df, df_job_encoded, df_marital_encoded, df_education_encoded], axis=1)
y_encoded = X_encoded['y']
X_encoded = X_encoded.drop('y', axis=1)
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Build and evaluate the model
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train_encoded, y_train_encoded)
y_pred = clf.predict(X_test_encoded)
print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
print("Precision:", precision_score(y_test_encoded, y_pred))
print("Recall:", recall_score(y_test_encoded, y_pred))
print("F1-score:", f1_score(y_test_encoded, y_pred))

# Visualize the decision tree and confusion matrix
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X_encoded.columns, class_names=["Not Subscribed", "Subscribed"], filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Classifier")
plt.show()
plt.figure(figsize=(10, 6))
disp = ConfusionMatrixDisplay.from_estimator(clf, X_test_encoded, y_test_encoded, display_labels=["Not Subscribed", "Subscribed"], cmap=plt.cm.Blues)
disp.ax_.set_title("Confusion Matrix")
plt.show()

# Hyperparameter tuning
param_dist = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
}
random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train_encoded, y_train_encoded)
best_params = random_search.best_params_
print("Best parameters found: ", best_params)
clf_best = DecisionTreeClassifier(**best_params)
clf_best.fit(X_train_encoded, y_train_encoded)
y_pred_best = clf_best.predict(X_test_encoded)
print("Optimized Accuracy:", accuracy_score(y_test_encoded, y_pred_best))
print("Optimized Precision:", precision_score(y_test_encoded, y_pred_best))
print("Optimized Recall:", recall_score(y_test_encoded, y_pred_best))
print("Optimized F1-score:", f1_score(y_test_encoded, y_pred_best))

# Visualize the optimized decision tree and confusion matrix
plt.figure(figsize=(20, 15))
plot_tree(clf_best, feature_names=X_train_encoded.columns, class_names=["Not Subscribed", "Subscribed"], filled=True, rounded=True, fontsize=8)
plt.title("Optimized Decision Tree Classifier")
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
disp = ConfusionMatrixDisplay.from_estimator(clf_best, X_test_encoded, y_test_encoded, display_labels=["Not Subscribed", "Subscribed"], cmap=plt.cm.Blues)
disp.ax_.set_title("Confusion Matrix - Optimized Model")
plt.show()

# Make predictions and visualize
example_customers = pd.DataFrame({
    'age': [30, 40, 50],
    'job_blue-collar': [0, 0, 1],
    'job_entrepreneur': [0, 1, 0],
    'job_housemaid': [0, 0, 0],
    'job_management': [0, 0, 0],
    'job_retired': [0, 0, 0],
    'job_self-employed': [0, 0, 0],
    'job_services': [0, 0, 0],
    'job_student': [0, 0, 0],
    'job_technician': [1, 0, 0],
    'job_unemployed': [0, 0, 0],
    'job_unknown': [0, 0, 0],
    'marital_married': [1, 0, 1],
    'marital_single': [0, 1, 0],
    'education_secondary': [1, 0, 1],
    'education_tertiary': [0, 1, 0],
    'education_unknown': [0, 0, 0]
})
predictions = clf_best.predict(example_customers)
predictions_proba = clf_best.predict_proba(example_customers)
for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
    print(f"Customer {i+1}:")
    print("Prediction (1 indicates Subscribed, 0 indicates not Subscribed):", pred)
    print("Prediction probability (probability of each class):", proba)
    print()
plt.figure(figsize=(8, 6))
sns.barplot(x=['Customer 1', 'Customer 2', 'Customer 3'], y=[proba[1] for proba in predictions_proba])
plt.title('Predicted Probability of Purchase for Example Customers')
plt.ylabel('Probability of Purchase')
plt.ylim(0, 1)


plt.show()

# Interpret the results
explainer = shap.TreeExplainer(clf_best)
shap_values = explainer.shap_values(X_test_encoded)
shap.summary_plot(shap_values[1], X_test_encoded, plot_type="bar", feature_names=X_encoded.columns)
shap.summary_plot(shap_values[1], X_test_encoded, feature_names=X_encoded.columns)
feature_importances = clf_best.feature_importances_
feature_names = X_encoded.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(importance_df)
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()
tree_rules = export_text(clf_best, feature_names=list(X_encoded.columns))
print("Decision Tree Rules:\n")
print(tree_rules)
dot_data = export_graphviz(clf_best, out_file=None, feature_names=X_encoded.columns, class_names=["Not Subcribed", "Subcribed"], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data, format="png")
graph.render("optimized_decision_tree", view=False, cleanup=True)
```

## Results
- The Decision Tree Classifier achieved an accuracy of approximately 88% on the test set.
- Hyperparameter tuning improved the model's performance with optimized parameters.
- Feature importance analysis identified the most influential features in predicting customer subscription.
- SHAP values provided insights into the decision-making process of the model.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
