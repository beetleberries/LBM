import numpy as np
import matplotlib.pyplot as plt

y = np.load("y_label_multi.npy")
unique, counts = np.unique(y, return_counts=True)

plt.figure(figsize=(6, 4))
plt.bar(unique, counts, tick_label=[f"Label {i}" for i in unique])
plt.title("Label Distribution")
plt.xlabel("Labels")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("label_distribution.png")
print(dict(zip(unique, counts)))
