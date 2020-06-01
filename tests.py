from optimization import Function
import numpy as np
from defineProblem import F

# 1d minimization
# f = x^2, x0 = 1, direction = [-1]
def f1var(X):
    return X[0]**2
def gradF1var(X):
    return 2*X[0]
F1var = Function(f1var)
print(F1var.minimizeDirection(np.array([1]),np.array([-1])))

# Check if point and lagrange multiplier is minimum
X = np.array([3,1])
l = np.array([2])
print(F.isMin(X,l))