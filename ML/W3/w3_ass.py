import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Load dataset
df = pd.read_csv("W3/advertising.csv")

X = df.drop("Sales", axis=1).values
y = df["Sales"].values.reshape(-1, 1)

# Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=16,
    shuffle=True
)

# Linear Regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel(X_train_t.shape[1])

# Loss and optimizer
# (Use MSE for training; compute RMSE with sklearn later)
# -----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Training
# -----------------------------
epochs = 200
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

# -----------------------------
# Evaluate with RMSE = sqrt(MSE)
# using: np.sqrt(metrics.mean_squared_error(y, yfit))
# -----------------------------
model.eval()
with torch.no_grad():
    y_train_fit = model(X_train_t).cpu().numpy()
    y_test_fit = model(X_test_t).cpu().numpy()

train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_fit))
test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_fit))

print("RMSE (Train) = ", train_rmse)
print("RMSE (Test)  = ", test_rmse)
