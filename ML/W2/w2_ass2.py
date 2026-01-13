from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load dataset
mnist = loadmat(r"W2\mnist-original.mat")  
# Prepare data
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]

# Select sample 32100
x = mnist_data[32100]
y = mnist_label[32100]

# Reshape and plot
x_image = x.reshape(28, 28)

plt.imshow(x_image, cmap="binary")
plt.title(f"Label: {int(y)}")
plt.axis("off")
plt.show()
