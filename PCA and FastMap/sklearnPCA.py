from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
data = np.loadtxt('pca-data.txt', dtype='float', delimiter="\t")
pca=PCA(n_components=2)
pca.fit(data)
print("sklearn result:")
result = pca.transform(data) 
for i in range(len(result)):
	print(result[i][0], result[i][1])
'''
for i in range(len(result)):
	plt.scatter(result[i][0], result[i][1])
plt.show()
'''