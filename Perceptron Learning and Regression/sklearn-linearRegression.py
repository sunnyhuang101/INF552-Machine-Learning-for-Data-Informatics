# INF552 HW4
# Shih-Yu Lai 
# Shiuan-Chin Huang 

from sklearn.linear_model import LinearRegression
from numpy import where
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def readFile():
    XY = np.array
    Z = np.array
    data = np.loadtxt("linear-regression.txt", dtype="float", delimiter=",")
    XY = np.array(data[:, 0:2]) # store X and Y, ex: [0.6937807956355748, 0.69754351093898]
    Z = np.array(data[:, 2]) # store Z,  ex: 3.2522896815114373, ...
    return XY, Z


def main():
    XY, Z = readFile()
    regr = LinearRegression()
    regr.fit(XY, Z)
    y_pred = regr.predict(XY)
    print("intercept : " + str(regr.intercept_))
    print("weights: " + str(regr.coef_))


if __name__ == "__main__":
    main()