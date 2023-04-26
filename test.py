import numpy as np

x = np.array([[0, 1, 2],
              [3, 4, 5]])
x1 = np.linalg.norm(x=x, ord=1, axis=0, keepdims=True)
x2 = np.linalg.norm(x=x, ord=1, axis=1, keepdims=True)
x3 = np.linalg.norm(x=x, ord=1, axis=0, keepdims=False)
x4 = np.linalg.norm(x=x, ord=2, axis=1, keepdims=False)

print(x.shape)
print(x1)
print(x2)
print(x3)
print(x4.shape)
