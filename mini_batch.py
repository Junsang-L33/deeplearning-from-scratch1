from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

(x_train, t_train), (x_test, t_test) = mnist.load_data()

#nomalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

#one_hot_label
t_train = to_categorical(t_train)
t_test = to_categorical(t_test)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

np.random.choice(60000, 10)
