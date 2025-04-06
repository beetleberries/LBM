import numpy as np
import matplotlib.pyplot as plt

# load data
y_preprocessed = np.load("y_label_3class.npy")
y_tokenized = np.load("y_label_3class_filtered.npy")

# mapping
label_names = {0: 'Drowsy', 1: 'Normal', 2: 'Distracted'}

def plot_distribution(y, title, filename):
    values, counts = np.unique(y, return_counts=True)
    labels = [label_names.get(v, str(v)) for v in values]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color='mediumslateblue')
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    for i, c in enumerate(counts):
        plt.text(i, c + 0.5, str(c), ha='center')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# visualize
plot_distribution(y_preprocessed, "Preprocessed Label Distribution", "label_dist_preprocessed.png")
plot_distribution(y_tokenized, "Tokenized Label Distribution", "label_dist_tokenized.png")
