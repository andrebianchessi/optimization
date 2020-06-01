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
    def __init__(self, name: str,
                 f: Callable[[np.ndarray], float] = None,
                 grad: Callable[[np.ndarray],np.ndarray] = None,
                 g: List[ Callable[[np.ndarray], float] ] = None,
                 gradG: List[ Callable[[np.ndarray], np.ndarray] ] = None):
        self.name  = name
        self.f     = f
        self.gradF = grad
        self.g     = g
        self.gradG = gradG
        self.H     = None
    
    def __str__(self):
        return self.name
    
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
                    print('Hessian matrix not positive definite!')
                    return False  
        return True
    
    def minimizeDirection(self, X0: np.ndarray, direction: np.ndarray) ->  np.ndarray:
        ''' Returns X that minimizes function in direction '''
        print('\nMinimize '+ self.name +' from '+ str(X0) +' in direction ', str(direction))
        def fDirection(n: float, *args):
            X0 = args[0]
            vector = args[1]
            return self.f(X0+n*vector)
        X1 = X0 + optimize.minimize(fDirection,np.array([0]),args=(X0,direction))['x']*direction
        print(" -> ", X1)
        return X1

    def minimizeGradientDescent(self, X0:np.ndarray, stopError: float) -> np.ndarray:
        direction = -1*self.gradF(X0)
        X1 = self.minimizeDirection(X0, direction)
        if np.linalg.norm(X1-X0) <= stopError:
            return X1
        else:
            return self.minimizeGradientDescent(X1, stopError)
    
    def minimizePowell(self, X0: np.ndarray, directions: List[np.ndarray], stopError: float) -> np.ndarray:
        x0 = X0
        x00 = X0
        for i in directions:
            x01 = self.minimizeDirection(x00, i)
            x00 = x01
        
        d = x01 - x0
        x1 = self.minimizeDirection(x0,d)

        if np.linalg.norm(x1-x0)<=stopError:
            return x1
        else:
            return self.minimizePowell(x1, directions, stopError)
    
    def minimizeExternalPenaltyGradientDescent(self, X0: np.ndarray, firstR: float, rFactor: float ,
                                               internalStopError: float, percentileStopError: float):
        '''
            Minimizes function using external penalty with increassing weights
            internalStopError: stop error for each weight r
            percentileStopError: global percentile error for varying weight r
        '''
        r = firstR
        def phi(X):
            s = 0
            for i in self.g:
                s = s + max(-i(X),0)**2
            return self.f(X) + r * s
        
        def gradPhi(X):
            s = 0
            for i in range(len(self.g)):
                if max(-self.g[i](X),0) >0:
                    s = s + 2 * self.g[i](X) * self.gradG[i](X)
            return self.gradF(X) + r*s
        
        Phi = Function('Penalty problem for ' + self.name, phi, gradPhi)

        x0 = Phi.minimizeGradientDescent(X0, internalStopError)
        r = r * rFactor
        x1 = Phi.minimizeGradientDescent(x0, internalStopError)

        while np.linalg.norm(x1-x0)/np.linalg.norm(x0) > percentileStopError:
            r = r * rFactor
            x0 = x1
            x1 = Phi.minimizeGradientDescent(x0, internalStopError)
        
        return x1
            




        return Phi.minimizeGradientDescent(X0, stopError)


        
        


        
            

        







        


