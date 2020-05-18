# INF552 HW4
# Shih-Yu Lai 
# Shiuan-Chin Huang 
import numpy as np
import random
import matplotlib.pyplot as plt

def readFile():
	X = np.array
	Y = np.array
	data = np.loadtxt("classification.txt", dtype="float", delimiter=",", usecols=(0,1,2,4))
	X = np.array(data[:, 0:3]) # [xi1, xi2, xi3] i = 1 to n
	Y = np.array(data[:, 3]) # -1 or 1
	return X, Y

def calErrors(result, ans):
	error = 0
	for i in range(len(result)):
		if result[i] != ans[i]:
			error += 1
	return error

def perception(w, X, Y):
	result = []

	for i in range(len(X)):
		wx = np.dot(w, X[i].T)
		yi = 1
		if wx <= 0:
			yi = -1
		
		result.append(yi)
	return result

def pocket(X,Y):
	plt_x = []
	plt_y = []
	w = np.zeros(4)
	result = perception(w, X, Y)
	min_error = calErrors(result, Y)
	best_w = w
	error = 0
	iteration = 0
	while iteration < 7000:

		rand = random.randint(0, len(X)-1)

		xi = X[rand]
		ans = Y[rand]
		wx = np.dot(w, xi.T)
		yi = 1
		if wx <= 0:
			yi = -1

		if yi != ans:
			iteration += 1
			mul = np.multiply(np.array([ans]), xi)
			w = np.add(w, mul)
			result = perception(w, X, Y)
			error = calErrors(result, Y)
			plt_y.append(error)
			plt_x.append(iteration)

			if error < min_error:
				best_w = w
				min_error = error
	print("weights w0, w1, w2, w3: %s" % best_w)
	print("accuracy: %.2f" % (100*((len(X)-min_error)/len(X))))
	plt.bar(plt_x, plt_y)
	plt.show()






def main():
	tmp, Y = readFile()
	ones = np.ones((1,len(tmp)))
	X = np.array
	X = np.concatenate((ones.T, tmp), axis=1)
	pocket(X, Y)

if __name__ == "__main__":
    main()