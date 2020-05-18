# INF552 HW4
# Shih-Yu Lai 
# Shiuan-Chin Huang

import numpy as np

def readFile():
    XY = np.array
    Z = np.array
    data = np.loadtxt("linear-regression.txt", dtype="float", delimiter=",")
    XY = np.array(data[:, 0:2]) # store X and Y, ex: [0.6937807956355748, 0.69754351093898]
    Z = np.array(data[:, 2]) # store Z,  ex: 3.2522896815114373, ...
    return XY, Z


def main():
    XY, Z = readFile()
    m, n = XY.shape # get the m*n number
    bias = np.ones((m, 1))
    XY = np.concatenate((bias, XY), axis = 1)
    XY_T = np.linalg.inv(np.dot(XY.T, XY))
    weights = np.dot(XY_T, np.dot(XY.T, Z))
    print("intercept : " + str(weights[0]))
    print("weights: " + str(weights[1:]))


if __name__ == "__main__":
    main()