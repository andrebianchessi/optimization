from optimization import Function
import numpy as np
from defineProblem import F1, F2, F3
print('\n\n')


# f = x^2, x0 = 1, direction = [-1]
# 1d minimization
def f1var(X):
    return X[0]**2
def gradF1var(X):
    return 2*X[0]
F1var = Function(f1var)
print(F1var.minimizeDirection(np.array([1]),np.array([-1])))


# Min       F = x^2 + y^2 -8x
# Such that G = y - x + 2 >=0
# Check if point and lagrange multiplier is minimum
X = np.array([3,1])
l = np.array([2])
print(F1.isMin(X,l))


# Min F = 8x^4 + 5y^4 + 2y^2 - 36x^2
# Gradient descent
X0 = np.array([-1,0])
e = 0.001
print(F2.minimizeGradientDescent(X0, e))


# Min F = 21*x^2 - 24*x*y + 30*x + 14*y^2-60*y
# Gradient descent
X0 = np.array([0,0])
e = 0.001
print(F3.minimizeGradientDescent(X0, e))
# Powell
print(F3.minimizePowell(X0,[np.array([1,0]),np.array([0,1])],e))


