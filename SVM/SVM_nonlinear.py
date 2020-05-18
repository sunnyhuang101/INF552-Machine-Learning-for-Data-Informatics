# INF552 HW6
# Shih-Yu Lai 
# Shiuan-Chin Huang 
# Dan-Hui Wu 
import numpy as np
from cvxopt import matrix, solvers
from math import pow
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
class SVM():
    def __init__(self):
        self.kernel_func = self.polynomial_kernel
        self.w = np.zeros(2)
        self.b = 0
        self.alphas = []
        self.alphaIdx = []

    def polynomial_kernel(self, x, y, c=5): #k(x,y) = (<x,y> + c)^2
        
        return pow(((x[0]*y[0]+x[1]*y[1])+c ), 2)

    def fit(self, X, L, C=1.5): # C: to prevent overfitting 
        self.QPsolver(X, L)
        
        
        
        #f(x) = sigma( ai*yi*k(x, xi) )
        for i in range(len(X)):
            self.b += L[i] 
            for m in range(len(self.alphas)):
                cur = self.alphaIdx[m]
                self.b = self.b -self.alphas[m]*L[cur]*self.kernel_func(X[i], X[cur])
        
        self.b = self.b/len(X)
        #print("=====")
        #print(self.b)

        #print(len(self.alphas))

    def QPsolver(self,  X, L, C=1.5):
        data_num = len(X)
        data_dim = len(X[0])
        P_tmp = np.ones((data_num, data_num))
        
        for i in range(data_num):
            #for j in range(data_num):
            P_tmp[i][i] = X[i][0]*X[i][0]+X[i][1]*X[i][1]

        #P:matrix, q: vector
        #print(P_tmp)
        P = 2*matrix(P_tmp)
        
        q = matrix(np.ones(data_num))

        
        #G, h: inequality constraints
        G_tmp = []
        for i in range(data_num):
            row = np.zeros(data_num).tolist()
            row[i] = -1.0
            G_tmp.append(row)

        for i in range(data_num):
            row = np.zeros(data_num).tolist()
            row[i] = 1.0
            G_tmp.append(row)
        
        G = matrix(np.asarray(G_tmp))
        
        tmps = []
        tmps = np.concatenate( (np.zeros(data_num).tolist(), (np.ones(data_num)*C).tolist()), axis=None ) 
        #print(tmps)
        h = matrix(tmps)
        
        #equality constraints
        #sigma(ai*yi) = 0
        A = matrix(L, (1, data_num))
        b = matrix(0.0)


        #quadratic programming: (1/2)*X.T*P*X + q.T*X, subject to Gx < h, Ax = b
        # 0 <= x <= C
        sol=solvers.qp(P, q, G, h, A, b)
        #print(sol['x'])
        for i in range(len(sol['x'])):
            if (sol['x'][i] > 1.5e-11): #2.4-11
                self.alphas.append(sol['x'][i])
                self.alphaIdx.append(i)
        #print(self.alphas)



        

def Draw(w, b, X, L, svm):
    
    for i in range(len(L)):
        
        if L[i] == 1.0:
            plt.scatter(float(X[i][0]), float(X[i][1]), c = "r") #+
        else:
            plt.scatter(float(X[i][0]), float(X[i][1]), c = "g") #-

    #w.T + b
    '''
    for i in range(len(svm.alphas)):
        idx = svm.alphaIdx[i]
        plt.scatter(float(X[idx][0]), float(X[idx][1]), c = "k")
    '''
    u = np.linspace(-20, 20,10)
    x, y = np.meshgrid(u,u)
    z = []

    for i in range(len(x)):
        for j in range(len(x[0])):
            sigma = 0
            for m in range(len(svm.alphas)):
                idx = svm.alphaIdx[m]
                x_cur = [x[i][j], y[i][j]]
                #w = sigma( ai*yi*phiXi*x )
                sigma += svm.alphas[m] * L[idx]*svm.kernel_func(x_cur, X[idx])
            z.append(sigma + svm.b)
   
    z = np.asarray(z).reshape((len(x), len(x[0])))
    
    plt.contour(x, y, z, colors='k') 


    
    plt.show()




def readFile():
    X = np.array
    L = np.array
    data = np.loadtxt("nonlinsep.txt", dtype="float", delimiter=",")
    X = np.array(data[:, 0:2]) # store x1 and x2, ex: [0.6937807956355748, 0.69754351093898]
    L = np.array(data[:, 2]) # store label,  ex: +1, -1
    return X, L

def main():
    X, L = readFile()
    svm = SVM()
    svm.fit(X, L)
    Draw(svm.w, svm.b, X, L, svm)
    print("%d Support Vectors" % (len(svm.alphas)))
    print("alphas:")
    print(svm.alphas)
    print("intercept:")
    print(svm.b)
if __name__ == "__main__":
    main()