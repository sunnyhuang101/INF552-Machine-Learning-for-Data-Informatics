# INF552 HW6
# Shih-Yu Lai 
# Shiuan-Chin Huang 
# Dan-Hui Wu 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
def main():
    data = np.loadtxt("linsep.txt",dtype = "float", delimiter=",")
    X = data[:,0:2]
    Y = data[:,2].reshape(-1,1)
    clf = SVC(C = 40, kernel = 'linear')
    clf.fit(X, Y.ravel())
    print('w = ',clf.coef_)
    print('b = ',clf.intercept_)
    print('Indices of support vectors = ', clf.support_)
    print('Support vectors = ', clf.support_vectors_)
    print('Number of support vectors for each class = ', clf.n_support_)
    print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))

    plt.scatter(X[:, 0], X[:, 1], c=data[:, 2], cmap='winter', alpha=1, s=50, edgecolors='r')
    x2_lefttargeth = -(clf.coef_[0][0] * (-1) + clf.intercept_) / clf.coef_[0][1]
    x2_righttargeth = -(clf.coef_[0][0] * (1) + clf.intercept_) / clf.coef_[0][1]
    plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], facecolors='none', s=100, edgecolors='k')
    plt.plot([-1, 1], [x2_lefttargeth, x2_righttargeth])
    plt.show()

if __name__ == "__main__":
	main()