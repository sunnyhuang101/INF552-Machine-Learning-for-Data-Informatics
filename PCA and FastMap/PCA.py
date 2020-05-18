import matplotlib.pyplot as plt
import numpy as np

#step0-subMean(): x' = x - mean
#step1: compute covariance matrix (n*n)
#step2: compute eigenvalue, eigenvector

def readFile():
    #data = np.loadtxt('pca-data.txt', dtype='float', delimiter="\t")
    #n*3 => 3*n, row major to column major

    data = []
    with open('pca-data.txt') as f:
    # with open('test.txt') as f:
        for line in f:
            line = line.split('\t')
            line_list = []
            for i in range(3):
            	line_list.append(float(line[i]))
            data.append(line_list)
    
    return data

def subMean(data):
	#mean = np.zeros(3)
	#mean = np.mean(data, axis=0)
	mean = [0.0, 0.0, 0.0]
	newData = []
	for i in range(len(data)):
		mean[0] = mean[0] + data[i][0]
		mean[1] = mean[1] + data[i][1]
		mean[2] = mean[2] + data[i][2]

	for i in range(3):
		mean[i] = mean[i]/len(data)
	

	#newData = data - mean
	for i in range(len(data)):
		rowData = []
		for j in range(3):
			rowData.append(data[i][j] - mean[j])
		newData.append(rowData)

	return newData,mean

def transpose(data):
	transposeData = []
	for i in range(3):
		tmp = []
		for j in range(len(data)):
			tmp.append(data[j][i])
		transposeData.append(tmp)

	return transposeData
#data: 3*n
#dataT: n*3
def calCov(data, dataT):
	'''
	dataT = []
	for i in range(3):
		tmp = []
		for j in range(len(data)):
			tmp.append(data[j][i])
		dataT.append(tmp)
	'''
	cov = []

	for i in range(len(data)):
		tmp = []
		for j in range(len(data)):
			total = 0
			for k in range(len(dataT)):
				total = total + data[i][k]*dataT[k][j]
			tmp.append(total)
		cov.append(tmp)

	for i in range(len(data)):
		for j in range(len(data)):
			cov[i][j] = cov[i][j]/len(dataT)
	
	return cov


def main():
	rdata = readFile()
	data, mean = subMean(rdata)
	transposeData = transpose(data)

	#cov = np.cov(data, rowvar=False)
	cov = calCov(transposeData, data)
	
	eigenVal, eigenVec = np.linalg.eig(cov)
	
	eigenDict = dict()
	for i in range(len(eigenVal)):
		eigenDict[i] = eigenVal[i]
	eigenDictSorted = sorted(eigenDict.items(), key=lambda item:item[1], reverse=True)

	eigenList = []
	count = 0
	for idx, eigenv in eigenDictSorted:
		eigenList.append(idx)
		count +=1 
		if count == 2:
			break

	newEigenVec = eigenVec[:, eigenList]
	newEigenVecT = newEigenVec.T
	newData = np.dot(newEigenVecT, transposeData)
	newData = np.asarray(newData)
	#result = data dot eigenvector
	result = np.dot(newEigenVecT, transposeData)
	result = result.T

	print("directions:")
	print(newEigenVecT)
	print("result:")
	for i in range(len(result)):
		print(result[i][0], result[i][1])
'''
	for i in range(len(result)):
		plt.scatter(result[i][0], result[i][1])
	plt.show()


	#rebuildData = np.dot(newData, newEigenVec.T) + mean
	rebuildData = np.dot(newData.T, newEigenVec.T) + np.asarray(mean)
	for i in range(len(data)):
		plt.scatter(float(data[i][0]), float(data[i][1]),float(data[i][2]), color='r')
		plt.scatter(float(rebuildData[i][0]), float(rebuildData[i][1]),float(rebuildData[i][2]), color='g')
		
	plt.show()
'''
	

if __name__ == "__main__":
    main()