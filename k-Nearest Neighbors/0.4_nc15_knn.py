import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

# ---------- Load & preprocess (done once) ----------
df1 = pd.read_csv('Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
df2 = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
df3 = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)
dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)

X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)
y = LabelEncoder().fit_transform(y)

# ---------- Settings ----------
N_COMPONENTS = 15
test_sizes = [ 0.4]
summary = []

# ---------- Loop over test sizes ----------
for ts in test_sizes:
    print("=" * 70)
    print(f"TEST SIZE = {ts}  |  PCA n_components = {N_COMPONENTS}")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, random_state=0, stratify=y
    )

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # PCA (unsupervised, no labels needed)
    pca = PCA(n_components=N_COMPONENTS, random_state=0)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"Variance explained by {N_COMPONENTS} components: {var_explained:.4f}\n")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"\nAccuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, zero_division=0))

    summary.append({'test_size': ts, 'n_components': N_COMPONENTS,
                    'variance_explained': var_explained,
                    'accuracy': acc, 'precision': prec,
                    'recall': rec, 'f1': f1})

# ---------- Summary table ----------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(pd.DataFrame(summary).round(4).to_string(index=False))