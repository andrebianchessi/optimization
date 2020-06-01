from optimization import Function
import numpy as np

# Define objective function
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

# Define restrictions
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


F = Function(f, gradF, g, gradG)
F.setH([[ddxddxf, ddxddyf], [ddyddxf, ddyddyf]])