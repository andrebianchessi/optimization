'''
    05/2020 - André Bianchessi
    andrebianchessi@gmail.com
'''

from typing import Callable, List
import numpy as np
from scipy import optimize
import scipy

class Function():
    '''
        Function to be minimized in standard KKT form,
        with k inequality constrains:

            min f
            g[1]  >= 0, g[2]  >= 0, ... , g[k]  >= 0

    '''
    def __init__(self, f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray],np.ndarray],
                 g: List[ Callable[[np.ndarray], float] ], gradG: List[ Callable[[np.ndarray], np.ndarray] ]):
        self.f     = f
        self.gradF  = grad
        self.g     = g
        self.gradG = gradG
        self.H     = None
    
    def constrainsOk(self, X: np.ndarray) -> bool:
        for i in self.g:
            if i(X) < 0:
                return False
        return True
    
    def KKTOk(self, X: np.ndarray, lambdas: np.ndarray) -> bool:
        # check inputs
        if len(lambdas) != len(self.g):
            raise Exception('Number of restrictions different than number of multiplers')

        # Viability
        if self.constrainsOk(X) != True :
            return False

        # Stationarity
        for i in lambdas:
            if i<0:
                return False
        
        s = 0
        for i in range(len(lambdas)):
            s = s + lambdas[i]*self.gradG[i](X)
        for i in (self.gradF(X) - s):
            if i != 0:
                return False
        
        # Complementarity
        for i in range(len(lambdas)):
            if lambdas[i] == 0 and not (self.g[i](X) != 0):
                return False
            if lambdas[i] != 0 and not (self.g[i](X) == 0):
                return False
        
        return True
    
    def setH(self, gradGradF: List[ List[ Callable[[np.ndarray], float] ]]) -> np.ndarray:
        '''
            gradGradF : [ [ d/dx1( d/dx1 f ), d/dx1( d/dx2 f ), ... , d/dx1( d/dxn f )  ] , 
                          [ d/dx2( d/dx1 f ), d/dx2( d/dx2 f ), ... , d/dx2( d/dxn f )  ] ,
                                                    ...                                   ,
                          [ d/dxn( d/dx1 f ), d/dxn( d/dx2 f ), ... , d/dxn( d/dxn f )  ] ]
        '''
        self.H = np.array(gradGradF)
    
    def getHEigenVals(self, X: np.ndarray) -> np.ndarray:
        H = np.copy(self.H)
        for i in range(len(self.H)):
            for j in range(len(self.H[i])):
                H[i][j] = H[i][j](X)
        H = np.array(H, dtype=float)
        return np.linalg.eig(H)[0]

    def isMin(self, X: np.ndarray, lambdas: np.ndarray) -> bool:
        '''
            Test KKT conditions and Hessian matrix
            If KKT conditions are met and Hessian matrix is positive
            definite, returns True
            Else, returns False
        '''    
        if self.KKTOk(X, lambdas):
            for i in self.getHEigenVals(X):
                if i<=0:
                    return False  
        return True
    
    def minimizeDirection(self, X0: np.ndarray, direction: np.ndarray) ->  scipy.optimize.OptimizeResult:
        def fDirection(n: float, *args):
            X0 = args[0]
            vector = args[1]
            return self.f(X0+n*vector)
        return optimize.minimize(fDirection,np.array([0]),args=(X0,direction))
        







        


