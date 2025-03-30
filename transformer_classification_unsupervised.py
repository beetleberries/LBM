import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

X = np.load("X_tokens_vqvae.npy")
y = np.load("y_label_multi.npy")

X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGTransformer(vocab_size=256, seq_len=X.shape[1], n_class=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
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

# Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        pred = model(xb).argmax(dim=1).cpu().numpy()
        y_true.extend(yb.numpy())
        y_pred.extend(pred)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
print("✅ Accuracy:", acc)
print("✅ F1-score:", f1)
print("\n", classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
