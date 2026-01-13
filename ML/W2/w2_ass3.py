from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# Load MNIST
mnist = loadmat(r"W2\mnist-original.mat")

X = mnist["data"].T
y = mnist["label"][0]

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Verify sizes
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
