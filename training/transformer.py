import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import math
import argparse


from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score # Added MSE, MAE, F1
from sklearn.metrics import roc_curve, auc, roc_auc_score # Added for ROC
from sklearn.preprocessing import label_binarize # Added for ROC
from itertools import cycle # Added for ROC plot colors
# --- Configuration ---
# Data Paths (MODIFY THESE)
EPOCHS_DATA_PATH = "D:/EEG/processed/stimlocked_psd_features_labeled_features.npy"
LABELS_PATH = "D:/EEG/processed/stimlocked_psd_features_labeled_labels.npy"

# Model Hyperparameters (Tune these!)
# Transformer
T_DIM = 128                 # Dimension for Transformer embeddings (d_model)
# INPUT_FEATURE_DIM will be n_channels from your data
T_NHEAD = 8                 # Number of attention heads
T_NUMLAYERS = 4             # Number of Transformer encoder layers
T_DIM_FEEDFORWARD = 256     # Dimension of feedforward layers in Transformer
T_DROPOUT = 0.2
T_LR = 1e-4
T_EPOCHS = 100 # Adjust based on convergence

# Training Hyperparameters
NUM_CLASSES = 3            # alert, transition, drowsy
BATCH_SIZE = 64
TEST_SPLIT_RATIO = 0.2 # For train/test split
VALIDATION_SPLIT_RATIO = 0.125 # Of the 80% training data, 12.5% for val (i.e. 10% of total)
RANDOM_SEED = 42

# Paths for saving models
T_MODEL_SAVE_PATH = './training/direct_transformer_classifier.pth'

# --- Global Lists for Plotting ---
tf_train_losses = []
tf_val_losses = []
tf_train_accuracies = []
tf_val_accuracies = []

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, embedding_dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- 2. Transformer Classifier Model ---
class DirectEEGTransformerClassifier(nn.Module):
    def __init__(self, input_feature_dim, model_dim, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1, max_seq_len=512):
        super(DirectEEGTransformerClassifier, self).__init__()
        self.model_dim = model_dim
        self.num_classes = num_classes

        # Project input features (e.g., from channels) to model_dim
        self.input_projection = nn.Linear(input_feature_dim, model_dim)

        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_len=max_seq_len + 1) # +1 for CLS

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.classifier_head = nn.Linear(model_dim, num_classes)

        # CLS token - learnable parameter
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, model_dim)) # (1, 1, Dim)

        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        self.classifier_head.bias.data.zero_()
        self.classifier_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: (Batch, Channels, SeqLen_Features) e.g. (64, 30, 100)
        # We want to treat SeqLen_Features as the sequence, and Channels as features per step.
        # So, permute to (Batch, SeqLen_Features, Channels)
        src = src.permute(0, 2, 1)  # (Batch, SeqLen_Features, Channels)

        # Project input features to model_dim
        projected_src = self.input_projection(src) # (Batch, SeqLen_Features, model_dim)
        batch_size = projected_src.shape[0]

        # Add CLS token embedding at the beginning of each sequence
        cls_tokens = self.cls_embedding.expand(batch_size, -1, -1) # (Batch, 1, model_dim)
        x = torch.cat((cls_tokens, projected_src), dim=1) # (Batch, 1 + SeqLen_Features, model_dim)

        # Add positional encoding - requires shape (1 + SeqLen_Features, Batch, model_dim)
        x = x.transpose(0, 1) # (1 + SeqLen_Features, Batch, model_dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # (Batch, 1 + SeqLen_Features, model_dim) - Ready for batch_first Transformer

        transformer_output = self.transformer_encoder(x) # (Batch, 1 + SeqLen_Features, model_dim)
        cls_output = transformer_output[:, 0, :] # (Batch, model_dim) - Use CLS token output
        logits = self.classifier_head(cls_output) # (Batch, Num Classes)
        return logits

# --- 3. Dataset Class ---
class EEGDataset(Dataset):
    def __init__(self, data, labels, label_map):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor([label_map[lbl] for lbl in labels], dtype=torch.long)
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# --- 4. Training and Evaluation Functions ---
def train_transformer(model, train_loader, val_loader, optimizer, criterion, epochs, device, scheduler=None):
    print("--- Starting Transformer Training ---")
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        current_train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device) # (Batch, Channels, SeqLen_Features)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            current_train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        avg_train_loss = current_train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_targets, train_preds)
        tf_train_losses.append(avg_train_loss)
        tf_train_accuracies.append(train_accuracy)

        # Validation Phase
        val_loss, val_accuracy, _, _ = evaluate_transformer(model, val_loader, criterion, device)
        tf_val_losses.append(val_loss)
        tf_val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss) # or val_accuracy
            else:
                scheduler.step()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"*** New best validation accuracy: {best_val_accuracy:.4f}. Saving model to {T_MODEL_SAVE_PATH} ***")
            torch.save(model.state_dict(), T_MODEL_SAVE_PATH)

    print("--- Transformer Training Finished ---")


def evaluate_transformer(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds_list = []
    all_targets_list = []
    all_probs_batches = [] # To store batches of probabilities

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data) # Logits
            loss = criterion(outputs, labels) # Ensure criterion is appropriate
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            probs = F.softmax(outputs, dim=1) # Calculate probabilities

            all_preds_list.extend(preds.cpu().numpy())
            all_targets_list.extend(labels.cpu().numpy())
            all_probs_batches.append(probs.cpu().numpy()) # Append batch of probabilities

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    all_targets_np = np.array(all_targets_list)
    all_preds_np = np.array(all_preds_list)

    if len(all_probs_batches) > 0:
        all_probs_np = np.concatenate(all_probs_batches, axis=0)
    else:
        # model.num_classes should exist from its constructor
        all_probs_np = np.empty((0, model.num_classes)) 
    
    accuracy = 0.0
    if len(all_targets_np) > 0: # Avoid error if dataloader was empty
        accuracy = accuracy_score(all_targets_np, all_preds_np)

    return avg_loss, accuracy, all_targets_np, all_preds_np, all_probs_np

# --- 5. Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Transformer directly on EEG features.')
    parser.add_argument('--load-transformer', action='store_true', help='Load pre-trained Transformer model.')
    args = parser.parse_args()

    print("Loading preprocessed data...")
    try:
        # Expects shape (num_samples, num_channels, num_features_per_channel_e.g_psd_bins)
        epochs_data = np.load(EPOCHS_DATA_PATH)
        labels_raw = np.load(LABELS_PATH, allow_pickle=True)
        print(f"Loaded epochs data shape: {epochs_data.shape}")
        print(f"Loaded labels shape: {labels_raw.shape}")

        n_samples, n_channels, n_seq_len_features = epochs_data.shape
        INPUT_FEATURE_DIM = n_channels # Features per step in the sequence will be channel values
                                       # Sequence length will be n_seq_len_features

    except FileNotFoundError:
        print(f"Error: Data files not found at {EPOCHS_DATA_PATH} or {LABELS_PATH}")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    unique_labels = sorted(list(set(labels_raw)))
    if len(unique_labels) != NUM_CLASSES:
        print(f"Warning: Found {len(unique_labels)} unique labels ({unique_labels}), but expected {NUM_CLASSES}. Adjusting NUM_CLASSES.")
        NUM_CLASSES = len(unique_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    reverse_label_map = {v: k for k, v in label_map.items()}
    print(f"Label Map: {label_map}")
    mapped_labels = np.array([label_map[lbl] for lbl in labels_raw])


    # Split data: First into Train and Test, then Train into Train_final and Validation
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        epochs_data,
        labels_raw, # Use raw string labels for stratification, will be mapped in Dataset
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels_raw
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SPLIT_RATIO, # e.g. 0.125 means 12.5% of (100-20)=80% data -> 10% of total for val
        random_state=RANDOM_SEED,
        stratify=y_train_val
    )

    print(f"Train data shape: {X_train.shape}, labels: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, labels: {y_val.shape}")
    print(f"Test data shape: {X_test.shape}, labels: {y_test.shape}")

    # Create Datasets
    train_dataset = EEGDataset(X_train, y_train, label_map)
    val_dataset = EEGDataset(X_val, y_val, label_map)
    test_dataset = EEGDataset(X_test, y_test, label_map)

    # --- Calculate Class Weights & Sampler for Imbalanced Data ---
    print("\nConfiguring for imbalanced data...")
    y_train_mapped = np.array([label_map[lbl] for lbl in y_train]) # Get integer labels for training set
    class_counts_train = np.bincount(y_train_mapped, minlength=NUM_CLASSES)

    # Option 1: WeightedRandomSampler
    sample_weights = np.array([1.0 / class_counts_train[lbl] if class_counts_train[lbl] > 0 else 0 for lbl in y_train_mapped])
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    train_sampler = WeightedRandomSampler(weights=sample_weights_tensor,
                                          num_samples=len(sample_weights_tensor),
                                          replacement=True)
    print(f"Training class counts: {class_counts_train}")
    print("Using WeightedRandomSampler for training loader.")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    # If not using sampler, shuffle train_loader:
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Option 2: Weighted CrossEntropyLoss (can be used with or without sampler)
    total_train_samples = len(y_train_mapped)
    class_weights_loss = total_train_samples / (NUM_CLASSES * class_counts_train + 1e-9)
    class_weights_loss_tensor = torch.tensor(class_weights_loss, dtype=torch.float).to(device)
    print(f"Calculated class weights for loss: {class_weights_loss_tensor.cpu().numpy()}")
    # criterion = nn.CrossEntropyLoss(weight=class_weights_loss_tensor) # Use this if preferred
    criterion = nn.CrossEntropyLoss() # Standard CE Loss if using sampler or no weighting

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # Initialize Transformer
    # max_seq_len for positional encoding should be at least n_seq_len_features
    transformer_model = DirectEEGTransformerClassifier(input_feature_dim=INPUT_FEATURE_DIM,
                                                       model_dim=T_DIM,
                                                       nhead=T_NHEAD,
                                                       num_layers=T_NUMLAYERS,
                                                       dim_feedforward=T_DIM_FEEDFORWARD,
                                                       num_classes=NUM_CLASSES,
                                                       dropout=T_DROPOUT,
                                                       max_seq_len=n_seq_len_features).to(device)

    optimizer = optim.AdamW(transformer_model.parameters(), lr=T_LR)
    # Optional: Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    scheduler = None


    if args.load_transformer:
        if os.path.exists(T_MODEL_SAVE_PATH):
            print(f"Loading pre-trained Transformer model from {T_MODEL_SAVE_PATH}")
            transformer_model.load_state_dict(torch.load(T_MODEL_SAVE_PATH, map_location=device))
            transformer_model.eval()
        else:
            print(f"Error: --load-transformer specified, but model file not found at {T_MODEL_SAVE_PATH}. Exiting.")
            exit()
    else:
        train_transformer(transformer_model, train_loader, val_loader, optimizer, criterion, T_EPOCHS, device, scheduler)

    


    # Final Evaluation on Test Set
    print("\n--- Evaluating Final Transformer Model on Test Set ---")
    # Load the best model saved during training for final eval
    try:
        # Ensure model is on the correct device before loading state_dict
        transformer_model.to(device) 
        transformer_model.load_state_dict(torch.load(T_MODEL_SAVE_PATH, map_location=device))
        print("Loaded best saved Transformer model state for final evaluation.")
    except FileNotFoundError:
        print(f"Warning: Best model checkpoint {T_MODEL_SAVE_PATH} not found. Evaluating with the current model state.")
    except Exception as e:
         print(f"Error loading best model checkpoint: {e}. Evaluating with current model state.")

    # Get predictions, probabilities, and targets from the test set
    test_loss, test_accuracy, test_targets, test_preds, test_probs = evaluate_transformer(
        transformer_model, test_loader, criterion, device
    )

    print(f"\n--- Test Set Performance Metrics ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # --- MSE and MAE (on class labels) ---
    # These metrics are typically for regression or ordinal classification.
    # For nominal classification, they compare integer class labels.
    mse = mean_squared_error(test_targets, test_preds)
    mae = mean_absolute_error(test_targets, test_preds)
    print(f"Test MSE (on class labels): {mse:.4f}")
    print(f"Test MAE (on class labels): {mae:.4f}")
    print("(Note: MSE and MAE on class labels are less common for nominal classification tasks.)")

    # --- F1 Scores (Macro, Weighted, Micro) ---
    # classification_report already provides many of these. This is for explicit access.
    # Ensure all NUM_CLASSES are considered for averaging
    f1_labels_to_consider = list(range(NUM_CLASSES))

    f1_macro = f1_score(test_targets, test_preds, average='macro', labels=f1_labels_to_consider, zero_division=0)
    f1_weighted = f1_score(test_targets, test_preds, average='weighted', labels=f1_labels_to_consider, zero_division=0)
    f1_micro = f1_score(test_targets, test_preds, average='micro', labels=f1_labels_to_consider, zero_division=0)
    # Note: Micro-F1 is equivalent to accuracy in multi-class classification.
    print(f"Test F1-Score (Macro): {f1_macro:.4f}")
    print(f"Test F1-Score (Weighted): {f1_weighted:.4f}")
    # print(f"Test F1-Score (Micro): {f1_micro:.4f}") # Optionally print, it's usually == accuracy

    # --- Classification Report and Confusion Matrix ---
    target_names = [reverse_label_map.get(i, f"Class {i}") for i in range(NUM_CLASSES)]
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=target_names, labels=f1_labels_to_consider, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_targets, test_preds, labels=f1_labels_to_consider)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Test Set (Direct Transformer)')
    plt.savefig('direct_transformer_confusion_matrix.png')
    plt.show(block=False) # Use block=False if other plots follow immediately

    # --- ROC Curve and AUC ---
    # Check if there are probabilities and more than one class for meaningful ROC
    if NUM_CLASSES > 1 and test_probs.shape[0] > 0 and test_probs.shape[1] == NUM_CLASSES:
        print("\n--- ROC Curve and AUC Analysis ---")
        y_test_binarized = label_binarize(test_targets, classes=list(range(NUM_CLASSES)))

        # Ensure y_test_binarized has the correct number of columns, even if some classes are not in test_targets
        if y_test_binarized.shape[1] != NUM_CLASSES:
             # This might happen if label_binarize does not create columns for absent classes.
             # Forcing it by creating a full matrix if necessary.
             temp_binarized = np.zeros((len(test_targets), NUM_CLASSES))
             for i, val in enumerate(test_targets):
                 if val < NUM_CLASSES : # Make sure label is within expected range
                     temp_binarized[i, val] = 1
             y_test_binarized = temp_binarized
             if y_test_binarized.shape[1] != NUM_CLASSES: # If still not matching, there's an issue
                  print(f"Warning: Binarized labels shape {y_test_binarized.shape} doesn't match NUM_CLASSES {NUM_CLASSES}. ROC might be incorrect.")


        fpr = dict()
        tpr = dict()
        roc_auc_values = dict() # Renamed from roc_auc to avoid conflict with auc function

        for i in range(NUM_CLASSES):
            if i < y_test_binarized.shape[1] and i < test_probs.shape[1]: # Check column exists
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], test_probs[:, i])
                roc_auc_values[i] = auc(fpr[i], tpr[i])
            else: # Should not happen if NUM_CLASSES is correct and data is consistent
                fpr[i], tpr[i], roc_auc_values[i] = np.array([0,1]), np.array([0,1]), 0.5
                print(f"Warning: Data for class {i} ({target_names[i]}) missing or inconsistent for ROC. Default AUC=0.5.")


        # Compute micro-average ROC curve and ROC area
        if y_test_binarized.size > 0 and test_probs.size > 0:
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), test_probs.ravel())
            if fpr["micro"].size > 0 and tpr["micro"].size > 0:
                roc_auc_values["micro"] = auc(fpr["micro"], tpr["micro"])
                print(f"Test ROC AUC (Micro-average): {roc_auc_values['micro']:.4f}")
        else:
            print("Warning: Cannot compute Micro-average ROC (empty binarized labels or probabilities).")

        # Compute macro-average ROC curve and ROC area
        # Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES) if i in fpr and fpr[i].size > 0]))
        
        if all_fpr.size > 0:
            mean_tpr = np.zeros_like(all_fpr)
            valid_classes_for_macro = 0
            for i in range(NUM_CLASSES):
                if i in fpr and fpr[i].size > 0 and tpr[i].size > 0:
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                    valid_classes_for_macro +=1
            
            if valid_classes_for_macro > 0:
                mean_tpr /= valid_classes_for_macro
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc_values["macro"] = auc(fpr["macro"], tpr["macro"])
                print(f"Test ROC AUC (Macro-average): {roc_auc_values['macro']:.4f}")
            else:
                print("Warning: Could not compute Macro-average ROC AUC (no valid per-class ROCs).")
        else:
             print("Warning: Could not compute Macro-average ROC AUC (all_fpr is empty).")

        # Sklearn's roc_auc_score for multi-class (OvR)
        try:
            if y_test_binarized.shape[0] > 0 and y_test_binarized.shape[1] == NUM_CLASSES:
                 # roc_auc_score handles cases where a class might not be present in y_true
                 # as long as y_score has probabilities for it.
                 sklearn_roc_auc_macro = roc_auc_score(y_test_binarized, test_probs, multi_class='ovr', average='macro')
                 sklearn_roc_auc_weighted = roc_auc_score(y_test_binarized, test_probs, multi_class='ovr', average='weighted')
                 print(f"Test ROC AUC (sklearn OVR Macro): {sklearn_roc_auc_macro:.4f}")
                 print(f"Test ROC AUC (sklearn OVR Weighted): {sklearn_roc_auc_weighted:.4f}")
            else:
                 print("Skipping sklearn roc_auc_score due to shape mismatch or empty data.")
        except ValueError as e:
            print(f"Could not compute sklearn roc_auc_score: {e}. This might be due to issues like a class having only one outcome in the test set.")

        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'limegreen', 'deeppink', 'indigo', 'gold']) # Add more if needed

        if "micro" in roc_auc_values and "micro" in fpr and "micro" in tpr:
            plt.plot(fpr["micro"], tpr["micro"],
                     label=f'Micro-average ROC (area = {roc_auc_values["micro"]:.2f})',
                     color='red', linestyle=':', linewidth=4)

        if "macro" in roc_auc_values and "macro" in fpr and "macro" in tpr:
            plt.plot(fpr["macro"], tpr["macro"],
                     label=f'Macro-average ROC (area = {roc_auc_values["macro"]:.2f})',
                     color='navy', linestyle=':', linewidth=4)

        for i, color in zip(range(NUM_CLASSES), colors):
            if i in roc_auc_values and i in fpr and i in tpr:
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'ROC curve of {target_names[i]} (area = {roc_auc_values[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2) # Dashed diagonal
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class Receiver Operating Characteristic (ROC) - Test Set')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('direct_transformer_roc_curves.png')
        plt.show(block=False) # Use block=False if other plots follow immediately

    elif test_probs.shape[0] == 0:
        print("\nROC Curves not plotted: No samples/probabilities from the test set evaluation.")
    elif test_probs.shape[1] != NUM_CLASSES:
        print(f"\nROC Curves not plotted: Mismatch between probability shape ({test_probs.shape[1]} classes) and NUM_CLASSES ({NUM_CLASSES}).")
    else: # NUM_CLASSES <= 1
        print(f"\nROC Curves not plotted: NUM_CLASSES is {NUM_CLASSES}. Meaningful ROC analysis requires at least 2 classes.")

    # Plotting Training Curves
    if not args.load_transformer and tf_train_accuracies: # Only plot if training was done
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(tf_train_losses, label='Train Loss')
        plt.plot(tf_val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Transformer Loss Curves')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(tf_train_accuracies, label='Train Accuracy')
        plt.plot(tf_val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Transformer Accuracy Curves')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('direct_transformer_training_plots.png')
        plt.show()

    print("\nScript finished.")