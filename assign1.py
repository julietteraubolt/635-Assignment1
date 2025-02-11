# Assignment 1: Implementing and comparing the performance 
# of KNN, Decision Trees, and Random Forests
# 
# You will:
# 1. Load and preprocess the dataset, including feature scaling for KNN.
# 2. Train and evaluate each model using metrics such as accuracy, precision, recall, and F1-
#    score.
# 3. Explore the impact of hyperparameter tuning on model performance.
# 4. Analyze and compare the models in a written report.
# 5. Submit your code via GitHub and a report summarizing your work.
# 
# Use the Breast Cancer dataset provided by sklearn. It includes 30 features and a binary
# classification task (malignant vs. benign).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Data preprocessing
#   Load the Breast Cancer dataset using load_breast_cancer from sklearn.
#   Partition the data into an 80% training set and a 20% test set.
#   Scale the features using StandardScaler for KNN.
data = load_breast_cancer()
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
x_train_scaled_KNN = scaler.fit_transform(x_train)
x_test_scaled_KNN = scaler.transform(x_test)

# Model training 
#   1. K-Nearest Neighbors (KNN): Start with n_neighbors=5.
#   2. Decision Tree: Use the default settings initially, then experiment
#      with max_depth.
#   3. Random Forest: Start with 100 trees (n_estimators=100) and
#      explore the effect of different max_depth or min_samples_split.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_scaled_KNN, y_train)
y_pred_knn = knn.predict(x_test_scaled_KNN)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

# Model evaluation
# Include: 
#   Accuracy
#   Precision
#   Recall
#   F1-score
# Include a confusion matrix for each model.
# Compare the results across the models in a tabular or graphical format.
print("K-Nearest Neighbors")
knn_accuracy = metrics.accuracy_score(y_test, y_pred_knn)*100
knn_precision = metrics.precision_score(y_test, y_pred_knn)*100
knn_recall = metrics.recall_score(y_test, y_pred_knn)*100
knn_f1 = metrics.f1_score(y_test, y_pred_knn)*100
print("Accuracy: {:.2f}%".format(knn_accuracy))
print("Precision: {:.2f}%".format(knn_precision))
print("Recall: {:.2f}%".format(knn_recall))
print("F1-score: {:.2f}%".format(knn_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

print("\nDecision Tree")
dt_accuracy = metrics.accuracy_score(y_test, y_pred_dt)*100
dt_precision = metrics.precision_score(y_test, y_pred_dt)*100
dt_recall = metrics.recall_score(y_test, y_pred_dt)*100
dt_f1 = metrics.f1_score(y_test, y_pred_dt)*100
print("Accuracy: {:.2f}%".format(dt_accuracy))
print("Precision: {:.2f}%".format(dt_precision))
print("Recall: {:.2f}%".format(dt_recall))
print("F1-score: {:.2f}%".format(dt_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

print("\nRandom Forest")
rf_accuracy = metrics.accuracy_score(y_test, y_pred_rf)*100
rf_precision = metrics.precision_score(y_test, y_pred_rf)*100
rf_recall = metrics.recall_score(y_test, y_pred_rf)*100
rf_f1 = metrics.f1_score(y_test, y_pred_rf)*100
print("Accuracy: {:.2f}%".format(rf_accuracy))
print("Precision: {:.2f}%".format(rf_precision))
print("Recall: {:.2f}%".format(rf_recall))
print("F1-score: {:.2f}%".format(rf_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Ablation study
# Modify key hyperparameters (e.g., n_neighbors for KNN, max_depth for
# Decision Trees and Random Forest) and observe the impact on
# performance.
print("\nAblation Study")

# Ablation study for K-Nearest Neighbors with differing 
# n_neighbors: 3, 7 (initially was 5)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
knn_3_accuracy = metrics.accuracy_score(y_test, y_pred_knn)*100
knn_3_precision = metrics.precision_score(y_test, y_pred_knn)*100
knn_3_recall = metrics.recall_score(y_test, y_pred_knn)*100
knn_3_f1 = metrics.f1_score(y_test, y_pred_knn)*100
print("K-Nearest Neighbors (n_neighbors=3)")
print("Accuracy: {:.2f}%".format(knn_3_accuracy))
print("Precision: {:.2f}%".format(knn_3_precision))
print("Recall: {:.2f}%".format(knn_3_recall))
print("F1-score: {:.2f}%".format(knn_3_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
knn_7_accuracy = metrics.accuracy_score(y_test, y_pred_knn)*100
knn_7_precision = metrics.precision_score(y_test, y_pred_knn)*100
knn_7_recall = metrics.recall_score(y_test, y_pred_knn)*100
knn_7_f1 = metrics.f1_score(y_test, y_pred_knn)*100
print("K-Nearest Neighbors (n_neighbors=7)")
print("Accuracy: {:.2f}%".format(knn_7_accuracy))
print("Precision: {:.2f}%".format(knn_7_precision))
print("Recall: {:.2f}%".format(knn_7_recall))
print("F1-score: {:.2f}%".format(knn_7_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# Ablation study for Decision Trees with differing 
# max_depth: 3, 5, 10 (initially was default None)
# It was at about this point that I had started 
# to wish I made a function for this, but we were 
# in too deep so here we are copying and pasting
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
dt_3_accuracy = metrics.accuracy_score(y_test, y_pred_dt)*100
dt_3_precision = metrics.precision_score(y_test, y_pred_dt)*100
dt_3_recall = metrics.recall_score(y_test, y_pred_dt)*100
dt_3_f1 = metrics.f1_score(y_test, y_pred_dt)*100
print("\nDecision Tree (max_depth=3)")
print("Accuracy: {:.2f}%".format(dt_3_accuracy))
print("Precision: {:.2f}%".format(dt_3_precision))
print("Recall: {:.2f}%".format(dt_3_recall))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
dt_5_accuracy = metrics.accuracy_score(y_test, y_pred_dt)*100
dt_5_precision = metrics.precision_score(y_test, y_pred_dt)*100
dt_5_recall = metrics.recall_score(y_test, y_pred_dt)*100
dt_5_f1 = metrics.f1_score(y_test, y_pred_dt)*100
print("\nDecision Tree (max_depth=5)")
print("Accuracy: {:.2f}%".format(dt_5_accuracy))
print("Precision: {:.2f}%".format(dt_5_precision))
print("Recall: {:.2f}%".format(dt_5_recall))
print("F1-score: {:.2f}%".format(dt_5_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

dt = DecisionTreeClassifier(max_depth=10)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
dt_10_accuracy = metrics.accuracy_score(y_test, y_pred_dt)*100
dt_10_precision = metrics.precision_score(y_test, y_pred_dt)*100
dt_10_recall = metrics.recall_score(y_test, y_pred_dt)*100
dt_10_f1 = metrics.f1_score(y_test, y_pred_dt)*100
print("\nDecision Tree (max_depth=10)")
print("Accuracy: {:.2f}%".format(dt_10_accuracy))
print("Precision: {:.2f}%".format(dt_10_precision))
print("Recall: {:.2f}%".format(dt_10_recall))
print("F1-score: {:.2f}%".format(dt_10_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# Ablation study for Random Forest with differing 
# max_depth: 3, 5, 10 (initially was default None)
# and then with min_samples_split: 2, 5, 10
rf = RandomForestClassifier(n_estimators=100, max_depth=3)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
rf_3_accuracy = metrics.accuracy_score(y_test, y_pred_rf)*100
rf_3_precision = metrics.precision_score(y_test, y_pred_rf)*100
rf_3_recall = metrics.recall_score(y_test, y_pred_rf)*100
rf_3_f1 = metrics.f1_score(y_test, y_pred_rf)*100
print("\nRandom Forest (max_depth=3)")
print("Accuracy: {:.2f}%".format(rf_3_accuracy))
print("Precision: {:.2f}%".format(rf_3_precision))
print("Recall: {:.2f}%".format(rf_3_recall))
print("F1-score: {:.2f}%".format(rf_3_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
rf_5_accuracy = metrics.accuracy_score(y_test, y_pred_rf)*100
rf_5_precision = metrics.precision_score(y_test, y_pred_rf)*100
rf_5_recall = metrics.recall_score(y_test, y_pred_rf)*100
rf_5_f1 = metrics.f1_score(y_test, y_pred_rf)*100
print("\nRandom Forest (max_depth=5)")
print("Accuracy: {:.2f}%".format(rf_5_accuracy))
print("Precision: {:.2f}%".format(rf_5_precision))
print("Recall: {:.2f}%".format(rf_5_recall))
print("F1-score: {:.2f}%".format(rf_5_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
rf_10_accuracy = metrics.accuracy_score(y_test, y_pred_rf)*100
rf_10_precision = metrics.precision_score(y_test, y_pred_rf)*100
rf_10_recall = metrics.recall_score(y_test, y_pred_rf)*100
rf_10_f1 = metrics.f1_score(y_test, y_pred_rf)*100
print("\nRandom Forest (max_depth=10)")
print("Accuracy: {:.2f}%".format(rf_10_accuracy))
print("Precision: {:.2f}%".format(rf_10_precision))
print("Recall: {:.2f}%".format(rf_10_recall))
print("F1-score: {:.2f}%".format(rf_10_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

rf = RandomForestClassifier(n_estimators=100, min_samples_split=2)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
rf_2_minss_accuracy = metrics.accuracy_score(y_test, y_pred_rf)*100
rf_2_minss_precision = metrics.precision_score(y_test, y_pred_rf)*100
rf_2_minss_recall = metrics.recall_score(y_test, y_pred_rf)*100
rf_2_minss_f1 = metrics.f1_score(y_test, y_pred_rf)*100
print("\nRandom Forest (min_samples_split=2)")
print("Accuracy: {:.2f}%".format(rf_2_minss_accuracy))
print("Precision: {:.2f}%".format(rf_2_minss_precision))
print("Recall: {:.2f}%".format(rf_2_minss_recall))
print("F1-score: {:.2f}%".format(rf_2_minss_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

rf = RandomForestClassifier(n_estimators=100, min_samples_split=5)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
rf_5_minss_accuracy = metrics.accuracy_score(y_test, y_pred_rf)*100
rf_5_minss_precision = metrics.precision_score(y_test, y_pred_rf)*100
rf_5_minss_recall = metrics.recall_score(y_test, y_pred_rf)*100
rf_5_minss_f1 = metrics.f1_score(y_test, y_pred_rf)*100
print("\nRandom Forest (min_samples_split=5)")
print("Accuracy: {:.2f}%".format(rf_5_minss_accuracy))
print("Precision: {:.2f}%".format(rf_5_minss_precision))
print("Recall: {:.2f}%".format(rf_5_minss_recall))
print("F1-score: {:.2f}%".format(rf_5_minss_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
rf_10_minss_accuracy = metrics.accuracy_score(y_test, y_pred_rf)*100
rf_10_minss_precision = metrics.precision_score(y_test, y_pred_rf)*100
rf_10_minss_recall = metrics.recall_score(y_test, y_pred_rf)*100
rf_10_minss_f1 = metrics.f1_score(y_test, y_pred_rf)*100
print("\nRandom Forest (min_samples_split=10)")
print("Accuracy: {:.2f}%".format(rf_10_minss_accuracy))
print("Precision: {:.2f}%".format(rf_10_minss_precision))
print("Recall: {:.2f}%".format(rf_10_minss_recall))
print("F1-score: {:.2f}%".format(rf_10_minss_f1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Combine ablation study results into a table and chart for comparison
models = [
    "KNN (k=5)", "KNN (k=3)", "KNN (k=7)",
    "Decision Tree (default)", "DT (max_depth=3)", "DT (max_depth=5)", "DT (max_depth=10)",
    "Random Forest (default)", "RF (max_depth=3)", "RF (max_depth=5)", "RF (max_depth=10)",
    "RF (min_samples_split=2)", "RF (min_samples_split=5)", "RF (min_samples_split=10)"
]

data = [
    [knn_accuracy, knn_precision, knn_recall, knn_f1],
    [knn_3_accuracy, knn_3_precision, knn_3_recall, knn_3_f1],
    [knn_7_accuracy, knn_7_precision, knn_7_recall, knn_7_f1],
    [dt_accuracy, dt_precision, dt_recall, dt_f1],
    [dt_3_accuracy, dt_3_precision, dt_3_recall, dt_3_f1],
    [dt_5_accuracy, dt_5_precision, dt_5_recall, dt_5_f1],
    [dt_10_accuracy, dt_10_precision, dt_10_recall, dt_10_f1],
    [rf_accuracy, rf_precision, rf_recall, rf_f1],
    [rf_3_accuracy, rf_3_precision, rf_3_recall, rf_3_f1],
    [rf_5_accuracy, rf_5_precision, rf_5_recall, rf_5_f1],
    [rf_10_accuracy, rf_10_precision, rf_10_recall, rf_10_f1],
    [rf_2_minss_accuracy, rf_2_minss_precision, rf_2_minss_recall, rf_2_minss_f1],
    [rf_5_minss_accuracy, rf_5_minss_precision, rf_5_minss_recall, rf_5_minss_f1],
    [rf_10_minss_accuracy, rf_10_minss_precision, rf_10_minss_recall, rf_10_minss_f1],
]

# Create DataFrame
df = pd.DataFrame(data, columns=["Accuracy", "Precision", "Recall", "F1-score"], index=models)
print(df)

# Plot bar charts for comparison
plt.figure(figsize=(12, 6))
df.plot(kind='bar', figsize=(14, 7), colormap='viridis', edgecolor='black')
plt.title("Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("Performance Metrics (%)")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Metrics")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

