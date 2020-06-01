from optimization import Function
import numpy as np
from defineProblems import F1var, F1, F2, F3
print('\n\n')
pageBreak = '\n\n-------------------------------------------------------------\n\n'


# f = x^2, x0 = 1, direction = [-1]
# 1d minimization
print(F1var.minimizeDirection(np.array([1]),np.array([-1])))
print(pageBreak)


# Min       F = x^2 + y^2 -8x
# Such that G = y - x + 2 >=0
# Check if point and lagrange multiplier is minimum
X = np.array([3,1])
l = np.array([2])
print('Point ',X, 'with multipliers ', l, 'is mimimum on F1: ',F1.isMin(X,l))
print(pageBreak)


# Min F = 8x^4 + 5y^4 + 2y^2 - 36x^2
# Gradient descent
X0 = np.array([-1,0])
e = 0.001
print('Min: ', F2.minimizeGradientDescent(X0, e))
print(pageBreak)


# Min F = 21*x^2 - 24*x*y + 30*x + 14*y^2-60*y
# Gradient descent
print('Gradient Descent')
X0 = np.array([100,20])
e = 0.001
print('Min: ', F3.minimizeGradientDescent(X0, e))
print(pageBreak)

# Powell
print('Powell')
print('Min: ', F3.minimizePowell(X0,[np.array([1,0]),np.array([0,1])],e))
print(pageBreak)


