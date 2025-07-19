import numpy as np

"""
a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)
"""

""" overflow problem
a = np.array([1010, 1000, 990])

np.exp(a) / np.sum(np.exp(a))

C = np.max(a)
a - C

np.exp(a - C) / np.sum(np.exp(a - C))
"""

def softmax(a):
  c = np.max(a)
  exp_a = np.exp(a - c) #overflow countermeasures
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y
