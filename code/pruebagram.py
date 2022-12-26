import numpy as np
from  trotelib import *

n = 4
m = 2
A = -np.eye(n)
A = A[:m,:]
print(A)
A2 = gram_schmidt(A)
print(A2)