# Introduction to Neural Networks

## 16. Softmax
```Python
import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    return [np.exp(i)/np.sum(np.exp(L)) for i in L]
```

## 21. Cross Entropy
```Python
import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.float64(Y)
    P = np.float64(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
```
