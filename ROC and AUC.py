import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Simulated data for demonstration
y_true = np.random.randint(0, 2, 10000)  # Binary labels (0 or 1)
y_scores = np.random.uniform(0, 1, 10000)  # Simulated predicted probabilities

# Compute ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="orange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance")  # Baseline

# Formatting the graph
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
plt.legend(loc="lower right")
plt.grid(True)

# Show the plot
plt.show()