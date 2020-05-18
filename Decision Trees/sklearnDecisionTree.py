from sklearn.tree import DecisionTreeClassifier
import re
import math
from collections import Counter, defaultdict
import datetime

keys = [] # list of attributes, label
trainData = [] #1-20  save each row as a dictionary
testData = [] #21-22
keysLen = 0 

label = "" #keys[KeysLen]
labelList = [] #1-20 values of label

attrValueDict = defaultdict(lambda:defaultdict())
attrValueDict["Occupied"]["High"] = 2
attrValueDict["Occupied"]["Moderate"] = 1
attrValueDict["Occupied"]["Low"] = 0
attrValueDict["Price"]["Expensive"] = 2
attrValueDict["Price"]["Normal"] = 1
attrValueDict["Price"]["Cheap"] = 0
attrValueDict["Music"]["Loud"] = 1
attrValueDict["Music"]["Quiet"] = 0
attrValueDict["Location"]["Talpiot"] = 4
attrValueDict["Location"]["City-Center"] = 3
attrValueDict["Location"]["German-Colony"] = 2
attrValueDict["Location"]["Ein-Karem"] = 1
attrValueDict["Location"]["Mahane-Yehuda"] = 0
attrValueDict["VIP"]["Yes"] = 1
attrValueDict["VIP"]["No"] = 0
attrValueDict["Favorite Beer"]["Yes"] = 1
attrValueDict["Favorite Beer"]["No"] = 0



def readFile():
	global keys, keysLen, label, labelList, trainData, testData, attributeList
	count = 1
	with open('dt_data.txt') as f:
		for line in f:
			if "(" in line: #first row, attributes and labels
				line = line.replace("\n", "").replace("(", "").replace(")", "")
				keys = line.split(", ")
				keysLen = len(keys)
				label = keys[keysLen-1]


			if any(char.isdigit() for char in line) == True:
				line = re.split("\d", line.replace("\n", "").replace(";", "").replace(":", ""))
				
				rowValues = line[2].strip().split(", ")
				rowData = []
				idx = 0

				for k in keys:

					if ( k == label):
						if rowValues[keysLen-1] == "No":
							labelList.append(0)
						else:
							labelList.append(1)
					else:
						rowData.append(attrValueDict[k][rowValues[idx]])
					idx += 1

				
				
				trainData.append(rowData)
				

def main():
	readFile()
	tree = DecisionTreeClassifier()
	tree = tree.fit(trainData, labelList)
	testData = []
	testData.append([1, 0, 1, 3, 0, 0])

	result = tree.predict(testData)

	if result[0] == 0:
		print("No")
	else:
		print("Yes")

if __name__ == "__main__":
	main()