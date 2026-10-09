import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

# ============================================================
# USER SETTINGS - change these as you wish
# ============================================================
N_COMPONENTS = 10      # LDA components (max = min(n_features, n_classes - 1))
TEST_SIZE    = 0.4     # fraction used for testing (e.g. 0.2 = 20%)
RANDOM_STATE = 0

HIDDEN_LAYERS = (64, 32)
EPOCHS        = 100     # max epochs
BATCH_SIZE    = 1024
LEARNING_RATE = 1e-3
PATIENCE      = 10      # early stopping patience (epochs without improvement)

FILES = [
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
]
# ============================================================

# ---------- Device ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
torch.manual_seed(42)
np.random.seed(42)

# ---------- Load data ----------
dataset = pd.concat([pd.read_csv(f, low_memory=True) for f in FILES], ignore_index=True)
dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
n_classes = len(labelencoder_y.classes_)

# ---------- Split first (avoids data leakage from the imputer) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ---------- LDA (runs on CPU, it's fast) ----------
max_components = min(X_train.shape[1], n_classes - 1)
n_comp = min(N_COMPONENTS, max_components)
if n_comp != N_COMPONENTS:
    print(f"N_COMPONENTS={N_COMPONENTS} is too large; LDA allows at most {max_components}. Using {n_comp}.")

lda = LinearDiscriminantAnalysis(n_components=n_comp)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Scale LDA output so the neural net trains well
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- Move data to GPU ----------
X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

# Hold out 10% of training data for early stopping
n_val = int(0.1 * len(X_train_t))
perm = torch.randperm(len(X_train_t), device=device)
val_idx, tr_idx = perm[:n_val], perm[n_val:]
X_val_t, y_val_t = X_train_t[val_idx], y_train_t[val_idx]
X_tr_t, y_tr_t = X_train_t[tr_idx], y_train_t[tr_idx]

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

# ---------- Model ----------
layers, in_dim = [], n_comp
for h in HIDDEN_LAYERS:
    layers += [nn.Linear(in_dim, h), nn.ReLU()]
    in_dim = h
layers.append(nn.Linear(in_dim, n_classes))
model = nn.Sequential(*layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------- Train ----------
best_val, best_state, wait = float('inf'), None, 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()
    print(f"Epoch {epoch:3d} | train loss {total / len(X_tr_t):.4f} | val loss {val_loss:.4f}")

    if val_loss < best_val - 1e-5:
        best_val, wait = val_loss, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping.")
            break

model.load_state_dict(best_state)

# ---------- Predict ----------
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).argmax(dim=1).cpu().numpy()

# ---------- Metrics ----------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n")
print(cm)

print("\nAccuracy : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nPrecision : {:.4f}".format(precision_score(y_test, y_pred, average='weighted', zero_division=0)))
print("\nRecall : {:.4f}".format(recall_score(y_test, y_pred, average='weighted', zero_division=0)))
print("\nF1 Score : {:.4f}".format(f1_score(y_test, y_pred, average='weighted', zero_division=0)))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))