from scipy.io import loadmat
import matplotlib.pyplot as plt
mnist = loadmat("mnist-original.mat")

print(mnist.keys())
print(mnist['data'].shape)
mnist_data = mnist["data"].T
print(mnist_data.shape)
mnist_label = mnist["label"][0]

#show image
x = mnist_data[20000]
x_image = x.reshape(28,28)

plt.imshow(x_image,cmap=plt.cm.binary,interpolation="nearest")
plt.show()
