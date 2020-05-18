# Group member : Shih-Yu Lai, Shiuan-Chin Huang
import re
import math
from collections import Counter, defaultdict

class TreeNode:
    def __init__(self, attribute, attrCounter):
        self.attribute = attribute
        # e.g. Occupied
        self.attrCounter = defaultdict(lambda: Counter())
        # e.g.
        # High : {Yes : 3, No : 2}
        # Moderate : {Yes : 5}
        self.children = defaultdict(lambda: True)
        # attribute value : has children nodes or not
        # e.g. High --> 3 Yes, 2 No ---> self.children[High] = True
        #	Moderate --> 5 Yes --> self.children[Moderate] = False
        self.label = defaultdict(lambda: "-")
        self.childrenTreeNodes = defaultdict(lambda: None)
        # attribute value: childTreeNode
        # e.g. occupied -> High : VIP
        # print (attribute)
        for v in attrCounter:
            for l in attrCounter[v]:
                self.attrCounter[v][l] = attrCounter[v][l]

            if len(self.attrCounter[v]) <= 1:  # finish
                self.children[v] = False
        # print(self.children[v])


keys = []  # list of attributes, label
trainData = []  # 1-22  save each row as a dictionary
testData = []
keysLen = 0

label = ""  # keys[KeysLen]
labelList = []  # 1-22 values of label
attributeList = []  # list of the six attributes
Mypredict = []
Draw = []


# attributeEntropy = dict(lambda: Counter)

def readFile():
    global keys, keysLen, label, labelList, trainData, testData, attributeList
    count = 1
    with open('dt_data.txt') as f:
        for line in f:
            if "(" in line:  # first row, attributes and labels
                line = line.replace("\n", "").replace("(", "").replace(")", "")
                keys = line.split(", ")
                keysLen = len(keys)
                label = keys[keysLen - 1]

            if any(char.isdigit() for char in line) == True:
                line = re.split("\d", line.replace("\n", "").replace(";", "").replace(":", ""))
                rowValues = line[2].split(", ")
                rowData = {}  # dictionary
                idx = 0

                for k in keys:
                    rowData[k] = rowValues[idx]
                    idx += 1

                    if (k == label):
                        labelList.append(rowValues[keysLen - 1])

                trainData.append(rowData)


                count += 1

    attributeList = keys[0:keysLen - 1]

def readTest():
    global keys, keysLen, label, labelList, trainData, testData, attributeList
    count = 1
    with open('dt_data_test.txt') as f:
        for line in f:
            if "(" in line:  # first row, attributes and labels
                line = line.replace("\n", "").replace("(", "").replace(")", "")
                keys = line.split(", ")
                keysLen = len(keys)
                label = keys[keysLen - 1]

            if any(char.isdigit() for char in line) == True:
                line = re.split("\d", line.replace("\n", "").replace(";", "").replace(":", ""))
                rowValues = line[2].split(", ")
                rowData = {}  # dictionary
                idx = 0

                for k in keys:
                    rowData[k] = rowValues[idx]
                    idx += 1

                testData.append(rowData)

    # print(testData)

def calculateRemainder(attrCounter):
    # print(attrCounter)
    remainder = 0
    totalDecisions = 0

    for v in attrCounter:
        for l in attrCounter[v]:
            totalDecisions += attrCounter[v][l]

    for v in attrCounter:  # values of an attribute
        # print(attrCounter[v])
        summation = 0
        entropy = 0
        for l in attrCounter[v]:
            summation += attrCounter[v][l]

        for l in attrCounter[v]:  # decision label of this value
            entropy -= attrCounter[v][l] / summation * math.log(attrCounter[v][l] / summation, 2)

        remainder += summation / totalDecisions * entropy

    return remainder


def calculateEntropy(attrValue, attrCounter):
    summation = 0
    entropy = 0

    for l in attrCounter[attrValue]:
        summation += attrCounter[attrValue][l]

    for l in attrCounter[attrValue]:  # decision label of this value
        entropy -= attrCounter[attrValue][l] / summation * math.log(attrCounter[attrValue][l] / summation, 2)

    return entropy


def calculateRootEntropy():  # label's entropy
    global labelList
    labelCounter = Counter(labelList)
    summation = len(labelList)
    entropy = 0

    for l in labelCounter:  # decision label of this value
        entropy -= labelCounter[l] / summation * math.log(labelCounter[l] / summation, 2)

    return entropy


# column major going through each attribute column, count remainders and recursively build decision tree
# maxAttrValues: the set of values of maxAtrr. e.g. maxAttr = "Occupied", maxAttrValues: {High, Moderate, Low}
def buildDecisionTree(parentEntropy, maxAttr, maxTreeNode, maxAttrValues, trainData):
    global attributeList, label, height

    if maxAttr != label:
        end = True
        for v in maxAttrValues:
            if maxTreeNode.children[v] == True:
                end = False
                break

        if end == True:  # if all the children nodes of this maxTreeNode are leaves
            return None

    maxEntropy = 0
    maxIG = 0
    parentAttr = maxAttr

    for a in attributeList:
        if a != parentAttr:
            attrCounter = defaultdict(lambda: Counter())
            # e.g. a = Occupied
            # High : {Yes : 3, No : 2}
            # Moderate : {Yes : 5}
            # Low : ....
            attrValues = []
            for col in range(len(trainData)):
                attrCounter[trainData[col][a]][trainData[col][label]] += 1
                attrValues.append(trainData[col][a])

            remainder = calculateRemainder(attrCounter)
            IG = parentEntropy - remainder

            if (IG > maxIG):
                maxIG = IG
                maxAttr = a
                maxTreeNode = TreeNode(a, attrCounter)
                maxAttrValues = set(attrValues)


    # print(maxAttrValues)
    # print(maxTreeNode.attrCounter)
    # print(maxAttr)
    for v in maxTreeNode.attrCounter:
        newTrainData = []

        for col in range(len(trainData)):
            if trainData[col][maxAttr] == v:
                newTrainData.append(trainData[col])
        # print(maxTreeNode.attribute)
        # print(v)
        # print(maxTreeNode.children[v])

        if maxTreeNode.children[v] == True:
            maxEntropy = calculateEntropy(v, maxTreeNode.attrCounter)
            maxTreeNode.childrenTreeNodes[v] = buildDecisionTree(maxEntropy, maxAttr, maxTreeNode, maxAttrValues, newTrainData)
        else:
            maxTreeNode.label[v] = newTrainData[0][label]

    return maxTreeNode


def predict(curTreeNode, testData):
    global Mypredict, Draw
    curAttrValue = testData[curTreeNode.attribute]
    print(curTreeNode.attribute + " ( " + curAttrValue + " ) -> ")

    while (curTreeNode.children[curAttrValue] == True):
        curTreeNode = curTreeNode.childrenTreeNodes[curAttrValue]
        curAttrValue = testData[curTreeNode.attribute]
        print(curTreeNode.attribute + " ( " + curAttrValue + " ) -> ")
    print("Enjoy : " + curTreeNode.label[curAttrValue])

    print("-------------------------------")

def main():
    readFile()
    global labelList, trainData, Mypredict
    # calculateEntropy(attributeList)
    parentEntropy = calculateRootEntropy()
    root = buildDecisionTree(parentEntropy, label, None, {}, trainData)

    readTest()
    for i in range(len(testData)):
        predict(root, testData[i])

if __name__ == "__main__":
    main()