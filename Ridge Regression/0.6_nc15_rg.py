import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

# ======================= CONFIG =======================
TEST_SIZES   = [ 0.6]   # edit as needed
N_COMPONENTS = 15                          # PCA components
RIDGE_ALPHA  = 1.0
RANDOM_STATE = 0
FILES = [
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
]
# ======================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}"
      + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))

# ---------- Load data (CPU, once) ----------
dataset = pd.concat([pd.read_csv(f, low_memory=False) for f in FILES], ignore_index=True)
dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.to_numpy(dtype=np.float64)

le = LabelEncoder()
y = le.fit_transform(dataset.iloc[:, -1].astype(str).str.strip())
n_classes = len(le.classes_)

n_comp = min(N_COMPONENTS, X.shape[1])
if n_comp != N_COMPONENTS:
    print(f"N_COMPONENTS={N_COMPONENTS} exceeds the number of features; using {n_comp}.")


# ---------- PCA on GPU ----------
def pca_fit(X, n_components):
    n = X.shape[0]
    mean = X.mean(0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (n - 1)
    evals, evecs = torch.linalg.eigh(cov)              # ascending order
    idx = torch.argsort(evals, descending=True)
    evals, evecs = evals[idx], evecs[:, idx]
    W = evecs[:, :n_components]
    var_ratio = evals[:n_components].sum() / evals.sum()
    return mean, W, var_ratio.item()


# ---------- Ridge classifier on GPU (closed form, one-vs-all targets) ----------
def ridge_fit(Z, y, n_classes, alpha):
    A = torch.cat([Z, torch.ones(len(Z), 1, dtype=Z.dtype, device=Z.device)], dim=1)
    Y = -torch.ones(len(Z), n_classes, dtype=Z.dtype, device=Z.device)
    Y[torch.arange(len(Z)), y] = 1.0
    reg = alpha * torch.eye(A.shape[1], dtype=Z.dtype, device=Z.device)
    reg[-1, -1] = 0.0                                  # don't penalise the bias
    return torch.linalg.solve(A.T @ A + reg, A.T @ Y)

def ridge_predict(Z, W):
    A = torch.cat([Z, torch.ones(len(Z), 1, dtype=Z.dtype, device=Z.device)], dim=1)
    return (A @ W).argmax(dim=1)


# ---------- Loop over test sizes ----------
summary = []

for ts in TEST_SIZES:
    print("\n" + "=" * 70)
    print(f"TEST SIZE = {ts}  |  PCA n_components = {n_comp}")
    print("=" * 70)

    # Split first (no test statistics leak into imputer/scaler/PCA)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, random_state=RANDOM_STATE, stratify=y)

    Xtr = torch.tensor(X_train, device=device)
    Xte = torch.tensor(X_test, device=device)
    ytr = torch.tensor(y_train, device=device, dtype=torch.long)

    # Mean imputation (train means only), then standardise (train stats only)
    col_mean = torch.nan_to_num(torch.nanmean(Xtr, dim=0), nan=0.0)
    Xtr = torch.where(torch.isnan(Xtr), col_mean, Xtr)
    Xte = torch.where(torch.isnan(Xte), col_mean, Xte)

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    # PCA (fit on train only)
    pca_mean, pca_W, var_ratio = pca_fit(Xtr, n_comp)
    Ztr = (Xtr - pca_mean) @ pca_W
    Zte = (Xte - pca_mean) @ pca_W
    print(f"Variance explained by {n_comp} components: {var_ratio:.4f}")

    # Ridge classifier
    W_ridge = ridge_fit(Ztr, ytr, n_classes, RIDGE_ALPHA)
    y_pred = ridge_predict(Zte, W_ridge).cpu().numpy()

    # Metrics (CPU)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    print(f"\nAccuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, labels=range(n_classes),
                                zero_division=0))

    summary.append({'test_size': ts, 'n_components': n_comp,
                    'variance_explained': var_ratio,
                    'accuracy': acc, 'precision': prec,
                    'recall': rec, 'f1': f1})

    # Free GPU memory between runs
    del Xtr, Xte, ytr, Ztr, Zte
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# ---------- Summary table ----------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(pd.DataFrame(summary).round(4).to_string(index=False))