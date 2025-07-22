from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

def img_show(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')  
    plt.show()

(x_train, t_train), (x_test, t_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
