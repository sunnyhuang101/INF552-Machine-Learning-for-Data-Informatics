# INF552 HW4
# Shih-Yu Lai 
# Shiuan-Chin Huang 

import numpy as np

data = []
acc = []

def readFile():
    X = np.array
    Y = np.array
    data = np.loadtxt("classification.txt", dtype="float", delimiter=",", usecols=(0,1,2,4))
    X = np.array(data[:, 0:3]) # store X
    Y = np.array(data[:, 3]) # store Y
    return X, Y

def train(X, Y, weights):
    Y = Y[:, np.newaxis]
    iteration = 0
    while iteration < 7000:
        s = np.multiply(np.dot(X, weights.T), Y)
        delta = (np.multiply(Y.T, X.T) / (1 + np.exp(s)).T).T
        Ein = np.sum(delta, axis=0)
        Ein = Ein / X.shape[0]
        weights += 0.001 * Ein
        iteration += 1

    return weights


def main():
    X, Y = readFile()
    m, n = X.shape  # get the m*n number
    bias = np.ones((m, 1))
    X = np.concatenate((bias, X), axis=1)
    weights = np.random.rand(1, n + 1) * 10
    weights = train(X, Y, weights)

    predict = np.dot(X, weights.T)
    ans = np.ones((m, 1));
    ans[predict < 0] = -1
    ans = np.multiply(ans, Y)
    ans[ans < 0] = 0
    acc.append(np.sum(ans, axis=0)[0] / float(m))
    print(np.sum(ans, axis=0)[0] / float(m))
    print(weights)

    # for i in range(100):
    #
    # print("------------------------------------")
    # print(min(acc))
    # print(max(acc))


if __name__ == "__main__":
    main()