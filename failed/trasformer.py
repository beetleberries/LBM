import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === 1. Load tokenized EEG sequence and filtered labels ===
X = np.load("X_tokens_vqvae.npy")  # (samples, sequence_len)
y = np.load("y_label_multi_filtered.npy")   # (samples,)

# === 2. TensorDataset 준비 ===
X_tensor = torch.tensor(X, dtype=torch.long)     # 정수 토큰
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)

# === 3. Train/Val Split ===
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

# === 4. Transformer Classifier ===
class EEGTransformerClassifier(nn.Module):
    def __init__(self, num_tokens, embed_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, embed_dim))  # max_len = 100

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

# === 5. 하이퍼파라미터 및 모델 선언 ===
model = EEGTransformerClassifier(
    num_tokens=256,      # VQ 코드북 크기
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    num_classes=5        # Fatigue, Attentive, Relaxed, Drowsy, Distracted
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === 6. 학습 루프 ===
epochs = 20
train_acc_list = []
val_acc_list = []
val_f1_list = []

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    acc = correct / total
    train_acc_list.append(acc)

    # === 검증 ===
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(y_val.cpu().numpy())

    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average='weighted')
    val_acc_list.append(val_acc)
    val_f1_list.append(val_f1)

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Train Acc: {acc*100:.2f}%")
    print(f"           Val Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")
    print(classification_report(val_targets, val_preds, digits=4))

# === 7. 시각화: 정확도와 F1 스코어 ===
plt.figure(figsize=(10, 5))
plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(val_acc_list, label="Val Accuracy")
plt.plot(val_f1_list, label="Val F1-score")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_progress.png")
plt.show()

# === 8. Confusion Matrix 시각화 및 저장 ===
cm = confusion_matrix(val_targets, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Validation)")
plt.savefig("confusion_matrix.png")
plt.show()
