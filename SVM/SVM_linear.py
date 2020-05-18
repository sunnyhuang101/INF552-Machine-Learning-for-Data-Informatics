# INF552 HW6
# Shih-Yu Lai 
# Shiuan-Chin Huang 
# Dan-Hui Wu 
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt
from sklearn.svm import SVC

class LinearSVM:
    def fit(self,X,Y):
        m,n = X.shape
        X_dash = Y*X
        H = np.dot(X_dash , X_dash.T) * 1.
       
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        G = cvxopt_matrix(-np.eye(m))
        h = cvxopt_matrix(np.zeros(m))
        A = cvxopt_matrix(Y.reshape(1, -1))
        b = cvxopt_matrix(np.zeros(1))
    
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol['x'])
        #Selecting the set of indices S corresponding to non zero parameters
        self.S = np.where(self.alphas>1e-5)[0]
        self.w = ((Y * self.alphas).T @ X).reshape(-1,1)
        self.b = Y[self.S] - np.dot(X[self.S], self.w)


def main():
    data = np.loadtxt("linsep.txt",dtype = "float", delimiter=",")
    X = data[:,0:2]
    Y = data[:,2].reshape(-1,1)
    m = LinearSVM()
    m.fit(X,Y)
    print("weights:",m.w)
    print("intercept:",m.b[0])
    print("support vectors:",X[m.S])

    plt.scatter(X[:, 0], X[:, 1], c=data[:, 2], cmap='winter', alpha=1, s=50, edgecolors='r')
    x2_lefttargeth = -(m.w[0] * (-1) + m.b[0]) / m.w[1]
    x2_righttargeth = -(m.w[0] * (1) + m.b[0]) / m.w[1]
    plt.scatter(X[m.S][:, 0], X[m.S][:, 1], facecolors='none', s=100, edgecolors='k')
    plt.plot([-1, 1], [x2_lefttargeth, x2_righttargeth])
    plt.show()

if __name__ == "__main__":
    main()
