from optimization import Function
import numpy as np

# f = x^2, x0 = 1, direction = [-1]
# 1d minimization
def f1var(X):
    return X[0]**2
def gradF1var(X):
    return 2*X[0]
F1var = Function('x^2',f1var)


# Min       F = x^2 + y^2 -8x
# Such that G = y - x + 2 >=0 
def f(X):
    x = X[0]
    y = X[1]
    return x**2 + y**2 -8*x
def gradF(X):
    x = X[0]
    y = X[1]
    return np.array( [2*x-8, 2*y] )
def ddxddxf(X):
    return 2
def ddxddyf(X):
    return 0
def ddyddxf(X):
    return 0
def ddyddyf(X):
    return 2
g     = []
gradG = []
def g1(X):
    x = X[0]
    y = X[1]
    return y-x+2
def gradG1(X):
    return np.array( [-1,1] )
g.append(g1)
gradG.append(gradG1)

F1 = Function('f1',f, gradF, g, gradG)
F1.setH([[ddxddxf, ddxddyf], [ddyddxf, ddyddyf]])

# Min F = 8x^4 + 5y^4 + 2y^2 - 36x^2
def f2(X):
    x = X[0]
    y = X[1]
    return 8*x**4 + 5*y**4 + 2*y**2 - 36*x**2
def gradF2(X):
    x = X[0]
    y = X[1]
    return np.array( [8*4*x**3+36*2*x, 5*4*y**3+2*2*y] )

F2 = Function('f2', f2, gradF2)

# Min F = 21*x^2 - 24*x*y + 30*x + 14*y^2-60*y 
def f3(X):
    x = X[0]
    y = X[1]
    return 21*x**2 - 24*x*y + 30*x + 14*y**2 - 60*y
def gradF3(X):
    x = X[0]
    y = X[1]
    return np.array( [21*2*x - 24*y + 30, -24*x + 14*2*y - 60] )

F3 = Function('f3',f3, gradF3)

