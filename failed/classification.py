import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load Tokenized Data 
X = np.load("X_tokens_vqvae_binary.npy")  # (samples, seq_len)
y = np.load("y_label_binary_filtered.npy")

X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# 2. Transformer Classifier
class EEGTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, n_class):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.pos = nn.Parameter(torch.randn(1, seq_len, 64))
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls = nn.Linear(64, n_class)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, :x.size(1), :]
        x = self.encoder(x)
        return self.cls(x.mean(dim=1))

# 3. Setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGTransformer(vocab_size=256, seq_len=X.shape[1], n_class=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Training
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"📚 Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. Evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        outputs = model(xb).argmax(dim=1).cpu().numpy()
        y_true.extend(yb.numpy())
        y_pred.extend(outputs)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion Matrix 
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fatigue"], yticklabels=["Normal", "Fatigue"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Metrics Bar Plot
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}

plt.figure(figsize=(6, 4))
bars = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="Blues")

# Show value
for i, bar in enumerate(bars.patches):
    value = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2f}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.ylim(0, 1.1)
plt.title("Evaluation Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("metrics_barplot.png")
plt.show()

# === 8. ROC Curve (Optional) ===
try:
    y_prob = []
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            probs = model(xb).softmax(dim=1)[:, 1]
            y_prob.extend(probs.cpu().numpy())

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()
except Exception as e:
    print(f"❗ ROC Curve Error: {e}")
