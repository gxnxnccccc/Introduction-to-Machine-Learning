import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay

# -----------------------------
# Load Iris
# -----------------------------
iris = datasets.load_iris()
X = iris["data"]          # (150, 4)
y = iris["target"]        # 0=setosa, 1=versicolor, 2=virginica
names = iris["target_names"]

# -----------------------------
# Select 2 classes only (binary)
# Example: setosa (0) vs versicolor (1)
# -----------------------------
class_a = 0  # setosa
class_b = 1  # versicolor

mask = (y == class_a) | (y == class_b)
X2 = X[mask]
y2 = y[mask]

# Convert labels to 0/1 for binary classification (optional but clearer)
# 0 -> class_a, 1 -> class_b
y_bin = (y2 == class_b).astype(int)

class_labels = [names[class_a], names[class_b]]

# -----------------------------
# Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X2, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# -----------------------------
# SGD Classifier
# -----------------------------
clf = SGDClassifier(random_state=42)
clf.fit(X_train, y_train)

# -----------------------------
# Predict + Confusion Matrix
# -----------------------------
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("\nAccuracy =", accuracy_score(y_test, y_pred) * 100, "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_labels))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.GnBu, values_format="d")
plt.title(f"SGDClassifier: {class_labels[0]} vs {class_labels[1]}")
plt.show()
