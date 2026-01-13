from sklearn import datasets

iris = datasets.load_iris()

X_slice = iris['data'][20:30]
y_slice = iris['target'][20:30]

print("Data:")
print(X_slice)

print("\nTarget:")
print(y_slice)

print("\nShape of data:", X_slice.shape)
