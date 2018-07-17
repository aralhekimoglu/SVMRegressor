import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SVR:
    def fit(self,X,y):
        self.X=X
        self.y=y
        self.n=X.shape[0]
        self.d=X.shape[0]
        
        self.Kernel=rbfKernel(X)
        self.supportValues=self.solveQP(eps=0.0001,C=1.)
        
        "Find w,w0"
        w=self.supportValues.T.dot(x)
        w0Array=y-x.dot(w)
        self.w0=w0Array.mean()
        
        return self
    def predict(self,x):
        predictionKernel=np.zeros( (self.n,1))
        for i in range (0,self.n):
            predictionKernel[i]=rbf(x,self.X[i])
        return self.supportValues.T.dot(predictionKernel)+self.w0
 
    def plotTest(self):
        x_test = np.array([i/10.0 for i in range (-15,15,1)]).reshape(30,1)
        y_test = np.array([self.predict(j) for j in x_test]).reshape(30,1)
        plt.plot(x_test, y_test, 'r')
        
    def solveQP(self,eps,C):
        "Calculate necessary matrices for qp"
        n=self.n
        P_part1 = np.concatenate((self.Kernel,np.zeros((n,n))))
        P = matrix(  np.concatenate((P_part1,np.zeros((2*n,n))),axis=1))        
        
        q=matrix(np.concatenate((-self.y,eps*np.ones((n,1))),axis=0))
        
        G=matrix(np.concatenate((np.eye(2*n),-np.eye(2*n)),axis=0))
        
        arrayOfCs=C*np.ones((n,1))
        h_part1=np.concatenate((arrayOfCs,2*arrayOfCs),axis=0)
        h_part2=np.concatenate((arrayOfCs,np.zeros((n,1)))  ,axis=0)
        h=matrix(np.concatenate((h_part1,h_part2),axis=0))
        
        A = matrix(np.concatenate((np.ones((n,1)),np.zeros((n,1))),axis=0), (1,2*n))
        b = matrix(0.0)
        sol=solvers.qp(P, q, G, h, A, b)
        supportValues=np.array( (sol['x'])[0:n] )
        return supportValues

def rbf(x1,x2,sigma=1):   
    return np.exp(np.linalg.norm(x1-x2)/(-2*sigma**2))

def rbfKernel(X):
    n=X.shape[0]
    Kernel=np.zeros((n,n))
    for i in range (0,n):
        for j in range (0,n):
            Kernel[i][j]=rbf(x[i],x[j])
    return Kernel

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
plt.plot(x, y, 'b.')

"Fit and predict from SVR"
svr=SVR()
svr.fit(x,y)
svr.plotTest()
y_pred=sc_y.inverse_transform(svr.predict(sc_x.transform(np.array([[6.5]]))))