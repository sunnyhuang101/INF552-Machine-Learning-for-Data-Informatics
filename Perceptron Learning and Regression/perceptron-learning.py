# INF552 HW4
# Shih-Yu Lai
# Shiuan-Chin Huang
import numpy as np

def readFile():
	X = np.array
	Y = np.array
	data = np.loadtxt("classification.txt", dtype="float", delimiter=",", usecols=(0,1,2,3))
	X = np.array(data[:, 0:3]) # [xi1, xi2, xi3] i = 1 to n
	Y = np.array(data[:, 3]) # -1 or 1
	return X, Y

def perception(X, Y):
	w = np.zeros(4)
	error = True
	misclass = 0

	while error == True:
		error = False
		misclass = 0
		for i in range(len(X)):
			wx = np.dot(w, X[i].T)
			yi = 1
			if wx <= 0:
				yi = -1

			if yi != Y[i]:
				error = True
				misclass += 1
				mul = np.multiply(np.array([Y[i]]), X[i])
				w = np.add(w, mul)
	print("weights w0, w1, w2, w3: %s" % w)
	print("accuracy: %.2f" % (100*((len(X)-misclass)/len(X))))



def main():
	tmp, Y = readFile()
	ones = np.ones((1,len(tmp)))
	X = np.array
	X = np.concatenate((ones.T, tmp), axis=1)
	perception(X, Y)




if __name__ == "__main__":
    main()