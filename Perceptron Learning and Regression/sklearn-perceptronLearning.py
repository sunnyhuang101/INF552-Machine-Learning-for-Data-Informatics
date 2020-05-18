# INF552 HW4
# Shih-Yu Lai 
# Shiuan-Chin Huang
from sklearn.linear_model import Perceptron
import numpy as np

def readFile():
	X = np.array
	Y = np.array
	data = np.loadtxt("classification.txt", dtype="float", delimiter=",", usecols=(0,1,2,3))
	X = np.array(data[:, 0:3]) # [xi1, xi2, xi3] i = 1 to n
	Y = np.array(data[:, 3]) # -1 or 1
	return X, Y


def perception(X, Y):
	clf = Perceptron(fit_intercept=False, max_iter=1300, shuffle=False)
	clf.fit(X, Y)
	print(clf.coef_)
	print(clf.score(X, Y))

def main():
	tmp, Y = readFile()
	ones = np.ones((1,len(tmp)))
	X = np.array
	X = np.concatenate((ones.T, tmp), axis=1)
	perception(X, Y)




if __name__ == "__main__":
    main()