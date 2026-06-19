# Import pandas for dataset handling
import pandas as pd

# Import matplotlib for graphs
import matplotlib.pyplot as plt

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import classifiers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Import joblib for saving files
import joblib


# ============================================
# LOAD DATASET
# ============================================

# Read diabetes dataset
data = pd.read_csv("diabetes.csv")


# ============================================
# FEATURE SELECTION
# ============================================

# Input features
X = data[[
    "Insulin",
    "Glucose",
    "BloodPressure",
    "BMI",
    "Age"
]]

# Target column
y = data["Outcome"]


# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================
# INITIALIZE MODELS
# ============================================

# Support Vector Machine
svm_model = SVC()

# KNN Classifier
knn_model = KNeighborsClassifier()

# Naive Bayes Classifier
nb_model = GaussianNB()


# ============================================
# TRAIN MODELS
# ============================================

# Train SVM
svm_model.fit(X_train, y_train)

# Train KNN
knn_model.fit(X_train, y_train)

# Train Naive Bayes
nb_model.fit(X_train, y_train)


# ============================================
# MAKE PREDICTIONS
# ============================================

# Predictions using SVM
svm_pred = svm_model.predict(X_test)

# Predictions using KNN
knn_pred = knn_model.predict(X_test)

# Predictions using Naive Bayes
nb_pred = nb_model.predict(X_test)


# ============================================
# CALCULATE SVM METRICS
# ============================================

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)


# ============================================
# CALCULATE KNN METRICS
# ============================================

knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)


# ============================================
# CALCULATE NAIVE BAYES METRICS
# ============================================

nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)


# ============================================
# PRINT RESULTS
# ============================================

print("\n===================================")
print("MODEL PERFORMANCE RESULTS")
print("===================================")

# ---------- SVM ----------
print("\n----- SVM -----")
print("Accuracy :", svm_accuracy)
print("Precision:", svm_precision)
print("Recall   :", svm_recall)
print("F1 Score :", svm_f1)

# ---------- KNN ----------
print("\n----- KNN -----")
print("Accuracy :", knn_accuracy)
print("Precision:", knn_precision)
print("Recall   :", knn_recall)
print("F1 Score :", knn_f1)

# ---------- NAIVE BAYES ----------
print("\n----- Naive Bayes -----")
print("Accuracy :", nb_accuracy)
print("Precision:", nb_precision)
print("Recall   :", nb_recall)
print("F1 Score :", nb_f1)


# ============================================
# SELECT BEST MODEL USING F1 SCORE
# ============================================

# Initially assume SVM is best
best_model = svm_model
best_model_name = "SVM"
best_accuracy = svm_accuracy
best_precision = svm_precision
best_recall = svm_recall
best_f1 = svm_f1

# Compare with KNN
if knn_f1 > best_f1:
    best_model = knn_model
    best_model_name = "KNN"
    best_accuracy = knn_accuracy
    best_precision = knn_precision
    best_recall = knn_recall
    best_f1 = knn_f1

# Compare with Naive Bayes
if nb_f1 > best_f1:
    best_model = nb_model
    best_model_name = "Naive Bayes"
    best_accuracy = nb_accuracy
    best_precision = nb_precision
    best_recall = nb_recall
    best_f1 = nb_f1


# ============================================
# STORE ALL METRICS
# ============================================

metrics = {

    "SVM": {
        "accuracy": float(svm_accuracy),
        "precision": float(svm_precision),
        "recall": float(svm_recall),
        "f1_score": float(svm_f1)
    },

    "KNN": {
        "accuracy": float(knn_accuracy),
        "precision": float(knn_precision),
        "recall": float(knn_recall),
        "f1_score": float(knn_f1)
    },

    "NaiveBayes": {
        "accuracy": float(nb_accuracy),
        "precision": float(nb_precision),
        "recall": float(nb_recall),
        "f1_score": float(nb_f1)
    },

    "BestModel": {
        "name": best_model_name,
        "accuracy": float(best_accuracy),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "f1_score": float(best_f1)
    }
}


# ============================================
# SAVE METRICS AND BEST MODEL
# ============================================

# Save all metrics
joblib.dump(metrics, "metrics.pkl")

# Save best model
joblib.dump(best_model, "model.pkl")

print("\nMetrics saved as metrics.pkl")
print("Best model saved as model.pkl")

print("\nMetrics saved successfully as metrics.pkl")


# ============================================
# DISPLAY BEST MODEL
# ============================================

print("\n===================================")
print("BEST MODEL")
print("===================================")

print("Best Model:", best_model_name)
print("Best F1 Score:", best_f1)


# ============================================
# GRAPH DATA
# ============================================

# Classifier names
models = ["SVM", "KNN", "Naive Bayes"]

# Accuracy values
accuracy_values = [
    svm_accuracy,
    knn_accuracy,
    nb_accuracy
]

# Precision values
precision_values = [
    svm_precision,
    knn_precision,
    nb_precision
]

# Recall values
recall_values = [
    svm_recall,
    knn_recall,
    nb_recall
]

# F1 Score values
f1_values = [
    svm_f1,
    knn_f1,
    nb_f1
]


# ============================================
# ACCURACY GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    models,
    accuracy_values,
    marker="o",
    linewidth=3
)

plt.title("Accuracy Comparison")
plt.xlabel("Classifiers")
plt.ylabel("Accuracy Score")
plt.grid(True)

# Save graph
plt.savefig("accuracy_graph.png")

# Show graph
plt.show()


# ============================================
# PRECISION GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    models,
    precision_values,
    marker="o",
    linewidth=3
)

plt.title("Precision Comparison")
plt.xlabel("Classifiers")
plt.ylabel("Precision Score")
plt.grid(True)

plt.savefig("precision_graph.png")

plt.show()


# ============================================
# RECALL GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    models,
    recall_values,
    marker="o",
    linewidth=3
)

plt.title("Recall Comparison")
plt.xlabel("Classifiers")
plt.ylabel("Recall Score")
plt.grid(True)

plt.savefig("recall_graph.png")

plt.show()


# ============================================
# F1 SCORE GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    models,
    f1_values,
    marker="o",
    linewidth=3
)

plt.title("F1 Score Comparison")
plt.xlabel("Classifiers")
plt.ylabel("F1 Score")
plt.grid(True)

plt.savefig("f1_score_graph.png")

plt.show()


# ============================================
# FINAL COMBINED COMPARISON GRAPH
# ============================================

plt.figure(figsize=(10, 6))

# Accuracy Line
plt.plot(
    models,
    accuracy_values,
    marker="o",
    linewidth=3,
    label="Accuracy"
)

# Precision Line
plt.plot(
    models,
    precision_values,
    marker="o",
    linewidth=3,
    label="Precision"
)

# Recall Line
plt.plot(
    models,
    recall_values,
    marker="o",
    linewidth=3,
    label="Recall"
)

# F1 Score Line
plt.plot(
    models,
    f1_values,
    marker="o",
    linewidth=3,
    label="F1 Score"
)

# Graph Title
plt.title("Classifier Performance Comparison")

# Axis Labels
plt.xlabel("Classifiers")
plt.ylabel("Performance Score")

# Grid
plt.grid(True)

# Legend
plt.legend()

# Save graph
plt.savefig("combined_comparison_graph.png")

# Show graph
plt.show()


# ============================================
# END OF PROJECT
# ============================================

print("\nAll graphs generated successfully.")
print("Project completed successfully.")