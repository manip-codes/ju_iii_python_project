import numpy as np
import pandas as pd
import torch

# =========================
# CUSTOM SETTINGS
# =========================

N_COMPONENTS = 5   # Change to 10 or 15
TEST_SIZE = 0.6     # Change to 0.1, 0.2, 0.3, etc.

# =========================
# GPU
# =========================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = torch.device("cuda")

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))

# =========================
# DATA
# =========================

df1 = pd.read_csv('Tuesday-WorkingHours.pcap_ISCX.csv', low_memory=True)
df2 = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv', low_memory=True)
df3 = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)

dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors='coerce')

X.replace([np.inf, -np.inf], np.nan, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=0,
    stratify=y
)

# =========================
# MOVE DATA TO GPU
# =========================

X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)

y_train = torch.tensor(y_train, dtype=torch.long, device=device)
y_test_gpu = torch.tensor(y_test, dtype=torch.long, device=device)

# =========================
# LDA - GPU
# =========================

classes = torch.unique(y_train)

mean = X_train.mean(dim=0)

Sw = torch.zeros(
    (X_train.shape[1], X_train.shape[1]),
    device=device
)

Sb = torch.zeros_like(Sw)

for c in classes:

    Xc = X_train[y_train == c]

    class_mean = Xc.mean(dim=0)

    centered = Xc - class_mean

    Sw += centered.T @ centered

    diff = (class_mean - mean).reshape(-1, 1)

    Sb += Xc.shape[0] * (diff @ diff.T)

Sw += 1e-4 * torch.eye(
    X_train.shape[1],
    device=device
)

M = torch.linalg.pinv(Sw) @ Sb

eigenvalues, eigenvectors = torch.linalg.eig(M)

eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

indices = torch.argsort(
    eigenvalues,
    descending=True
)

W = eigenvectors[:, indices[:N_COMPONENTS]]

X_train = X_train @ W
X_test = X_test @ W

# =========================
# LOGISTIC REGRESSION - GPU
# =========================

num_classes = len(classes)

classifier = torch.nn.Linear(
    N_COMPONENTS,
    num_classes
).to(device)

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=0.01
)

loss_function = torch.nn.CrossEntropyLoss()

for epoch in range(300):

    optimizer.zero_grad()

    output = classifier(X_train)

    loss = loss_function(
        output,
        y_train
    )

    loss.backward()

    optimizer.step()

# =========================
# PREDICTION
# =========================

with torch.no_grad():

    y_pred = torch.argmax(
        classifier(X_test),
        dim=1
    )

# Move only results to CPU
y_pred = y_pred.cpu().numpy()
y_test = y_test_gpu.cpu().numpy()

# =========================
# METRICS
# =========================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)

print("\nN Components:", N_COMPONENTS)
print("Test Size:", TEST_SIZE)

print("\nConfusion Matrix:\n")
print(cm)

print("\nAccuracy : {:.4f}".format(
    accuracy_score(y_test, y_pred)
))

print("\nPrecision : {:.4f}".format(
    precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )
))

print("\nRecall : {:.4f}".format(
    recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )
))

print("\nF1 Score : {:.4f}".format(
    f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )
))

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))