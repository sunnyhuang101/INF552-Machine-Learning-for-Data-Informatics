# INF552 HW4
# Shih-Yu Lai
# Shiuan-Chin Huang 

from sklearn.linear_model import LogisticRegression
from numpy import where
import numpy as np

def readFile():
    X = np.array
    Y = np.array
    data = np.loadtxt("classification.txt", dtype="float", delimiter=",", usecols=(0,1,2,4))
    X = np.array(data[:, 0:3]) # store coordinates of a point
    Y = np.array(data[:, 3]) # store label
    return X, Y

def main():
    X, Y = readFile()
    regr = LogisticRegression()
    regr.fit(X, Y)
    y_pred = regr.predict(X)

    correct = where(Y == y_pred)[0].shape[0]
    total = y_pred.shape[0]
    acc = correct / total
    print("Accuracy: ", acc)
    print("Weights: ", regr.coef_)

if __name__ == "__main__":
    main()