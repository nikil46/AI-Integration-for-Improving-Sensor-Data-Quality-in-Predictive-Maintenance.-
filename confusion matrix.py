import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define confusion matrix values
conf_matrix = np.array([[55456, 886], 
                        [4949, 51411]])

# Define class labels
labels = ["0", "1"]

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="magma", xticklabels=labels, yticklabels=labels)

# Add labels and title

plt.title("Confusion Matrix")

# Show plot
plt.show()