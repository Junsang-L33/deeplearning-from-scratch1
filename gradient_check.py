import numpy as np
from TwoLayerNet_useLayer import TwoLayerNet
rom tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_mnist(normalize=True, one_hot_label=True):
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    if one_hot_label:
        t_train = np.eye(10)[t_train]
        t_test = np.eye(10)[t_test]

    return (x_train, t_train), (x_test, t_test)



(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
  diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
  print(key + ":" + str(diff))
