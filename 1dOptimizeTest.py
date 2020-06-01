from optimization import Function
import numpy as np

def f(X):
    return X[0]**2
def gradF(X):
    return 2*X[0]

F = Function(f,None,None,None)

print(F.minimizeDirection(np.array([1]),np.array([-1])))
