# INF552 HW2
# Shih-Yu Lai 
# Shiuan-Chin Huang 
import matplotlib.pyplot as plt
import numpy as np
import Kmeans as km
from collections import Counter, defaultdict

#Kmeans.readFile()

K = 3
data = [] #store the input data from Kmeans.py
cls = [] #store the cls from Kmeans.py
MAX_ITER = 50
clsKData = defaultdict(list)


def Expectation(firstRun, gammas, dataArray):
	global K, cls, data, clsKData
	N = len(data)
	
	means = []
	covs = []
	amplitudes = []

	#firstRun: use the results from Kmeans.py to calculate 
	if firstRun == True:
		for k in range(K):
			means.append(np.mean(np.array(clsKData[k]), axis = 0).tolist())
			amplitudes.append(len(clsKData[k])/len(data))
			newCov = np.zeros((2,2))

			for i in range(len(clsKData[k])):
				cov1 = np.reshape(np.subtract(np.array(clsKData[k][i]), means[k]), (1,2))
				cov1T = np.reshape(cov1, (2,1))
				newCov += np.dot(cov1T, cov1)
				#newCov += np.dot(np.subtract(np.array(clsKData[k][i]), np.array(means[k])).T, np.subtract(np.array(clsKData[k][i]), np.array(means[k])))

			covs.append((newCov/len(clsKData[k])).tolist())
		

	else:
		Nk = []
		for k in range(K):
			x = np.zeros((1,2))

			Nk.append(np.sum(gammas[:, k]))
			amplitudes.append(Nk[k]/N)
			for i in range(len(dataArray)):
				x[0] = np.add(x[0], np.multiply(dataArray[i], gammas[i][k]))
				
				#dataIdx[k].append(i)
				#clsKDataMean[k].append([np.multiply(float(data[i][0]), gammas[i][k]), np.multiply(float(data[i][1]), gammas[i][k])])
			means.append((x[0]/Nk[k]).tolist())
		

		for k in range(K):
			newCov = np.zeros((2,2))
			for i in range(N):
				cov1 = np.reshape(np.subtract(dataArray[i], means[k]), (1,2))
				cov1T = np.reshape(cov1, (2,1))
				newCov += gammas[i][k]*np.dot(cov1T, cov1)
			
			
				
			covs.append((newCov/Nk[k]).tolist())
		#print(covs)
			

	return means, covs, amplitudes		
		



def Maximization(means, covs, amplitudes, dataArray, firstRun):
	global K, cls, data
	gammas = np.zeros((len(data), 3)) #possibility of data i be sorted to cluster k
	
	N = len(data)
	pb = np.zeros((len(data), 3)) #N(x|mean, cov) = possibility density function


	for k in range(K):
		for i in range(len(data)):
			a = np.subtract(dataArray[i], np.array(means[k])) #(x - mean)
			b = a.T #(x - mean)T
			#print(covs[k])
			x = 0

			#if (firstRun == True):
			#x = 1/(2*np.pi*pow(np.abs(covs[k]), 0.5))*np.exp(-0.5*(np.dot(a*1/(covs[k]), b)))
			#else:
			#print(covs[k])
			x = 1/(2*np.pi*pow(np.linalg.det(np.array(covs[k]).reshape(2,2)), 0.5))*np.exp(-0.5*(np.dot(np.dot(a, np.linalg.inv(np.array(covs[k]).reshape(2,2))), b)))
			#print(x)
			pb[i][k] = x

	#sumAmplitudePb = [] #summation of amplitudes*pb
	for i in range(N):
		sumAmplitudePb = 0 # summation of amplitudes[k]*pb[i][k] for each data i of cluster 1-3
		for k in range(K):
			sumAmplitudePb += amplitudes[k]*pb[i][k]


		for k in range(K):
			gammas[i][k] = amplitudes[k]*pb[i][k]/sumAmplitudePb

	#print(gammas)
	return gammas



def main():
	global cls, data, clsKData
	km.main()
	data = km.data
	cls = km.cls
	N = len(data)
	dataArray  = np.zeros((len(data),2)) #2-dim array for all data

	for i in range(N):
		k = cls[i]
		clsKData[k].append([float(data[i][0]), float(data[i][1])]) #for first time calculation of mean, amplitude, cov
		dataArray[i][0] = float(data[i][0]) 
		dataArray[i][1] = float(data[i][1])

	
	gammas = np.zeros((len(data), 3)) #possibility of data i to cluster k
	means, covs, amplitudes = Expectation(True, gammas, dataArray)
	time = 0
	converge = False
	gammas = Maximization(means, covs, amplitudes, dataArray, True)
		

	while time < MAX_ITER and converge == False:
		prevMeans = means
		means, covs, amplitudes = Expectation(False, gammas, dataArray)
		gammas = Maximization(means, covs, amplitudes, dataArray, False)
		time += 1
		meanDiff = np.abs(np.subtract(means, prevMeans))
		converge = True

		for i in range(3):
			#print(meanDiff[i])
			meanDis = pow((meanDiff[i][0]**2 + meanDiff[i][1]**2), 0.5)
			#print(meanDis)
			if (meanDis >= 0.00001):
				converge = False

	for i in range(3):
		print("Mean",i+1, ": ")
		print(means[i])
		print("Covariance" ,i+1, ": ")
		for j in range(2):
			print(covs[i][j])
		print("Amplitude" ,i+1, ": ")
		print(amplitudes[i])


	

	
    

if __name__ == "__main__":
    main()