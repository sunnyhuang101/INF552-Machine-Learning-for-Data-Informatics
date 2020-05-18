# INF552 HW3
# Shih-Yu Lai 
# Shiuan-Chin Huang 

import matplotlib.pyplot as plt
import numpy as np
import random

k = 2 # Dimension number
listObj = [] # Store all the objects number, in this case, is 1 ~ 10
maxObj = [] # [0]: ObjA, [1]: ObjB, [2]: maxDistance
result = np.zeros((10, k)) # The final answer

# Read 2D points and word-list from txt and store in numpy array
def readFile():
    Obj = np.array
    dis = np.array
    data = np.loadtxt('fastmap-data.txt', dtype='float', delimiter="\t")
    Obj = np.array(data[:, 0:2]) # store two objects ID, ex: [1,2] [1,3] ...
    dis = np.array(data[:, 2]) # store the distance of the objects ex: 4,7,6 ...
    label = np.loadtxt('fastmap-wordlist.txt', dtype='str')
    return Obj, dis, label

# Store all the objects number, listObj is 1 ~ 10
def List(Obj):
    for i in range(len(Obj)):
        if(Obj[i][0] not in listObj):
            listObj.append(Obj[i][0])
        if (Obj[i][1] not in listObj):
            listObj.append(Obj[i][1])

# Find the farthest object from ObjB and max distance
def FindMaxDis(Obj, dis, ObjB):
    Far = 0
    maxDis = 0
    Rand = []
    for idx, myObj in enumerate(Obj):
        if myObj[0] == ObjB:
            if dis[idx] >= maxDis:
                maxDis = dis[idx]
                Far = myObj[1]
        elif myObj[1] == ObjB:
            if dis[idx] >= maxDis:
                maxDis = dis[idx]
                Far = myObj[0]

    return Far, maxDis

# Randomly choose one object become ObjB, then find the ObjA and maxDistance
def findObj(Obj, dis, iter):
    ObjA = 0
    org = 0
    ObjB = Obj[random.randrange(len(listObj))][0]
    maxDis = 0
    preDis = 0
    FindMax = False

    if iter == 0:
        while FindMax != True:
            ObjA, maxDis = FindMaxDis(Obj, dis, ObjB)
            if(maxDis > preDis and org != ObjA and maxDis != 0):
                org = ObjB
                ObjB = ObjA
                preDis = maxDis
            else:
                ObjA = org
                FindMax = True
                maxDis = preDis
    else:
        ObjA = maxObj[0]
        ObjB = maxObj[1]
        maxDis = maxObj[2]

    return ObjA, ObjB, maxDis

# First dimension projection
# Store the first project points of all the objects
def project_xi(Obj, dis, ObjA, ObjB, dab, iter):
    Xi = 0
    dai = 0
    dbi = 0

    for idx1, Oi in enumerate(listObj):
        for idx2, myObj in enumerate(Obj):
            if ObjA == Oi:
                dai = 0
            elif (myObj[0] == ObjA and myObj[1] == Oi) or (myObj[1] == ObjA and myObj[0] == Oi):
                dai = dis[idx2]

            if ObjB == Oi:
                dbi = 0
            elif (myObj[0] == ObjB and myObj[1] == Oi) or (myObj[1] == ObjB and myObj[0] == Oi):
                dbi = dis[idx2]

        Xi = (dai**2 + dab**2 - dbi**2) / ( 2*dab )
        result[idx1][iter] = Xi

# Second dimension projection
# Store the second project points of all the objects
def project_k_dimention(Obj, dis, iter):
    newDis = np.zeros((len(dis), 1))
    dOij = 0
    dXij = 0.0
    maxDis = 0
    maxObjA = 0
    maxObjB = 0

    for idx1, Oj in enumerate(Obj):
        for idx2, myObj in enumerate(Obj):
            if Oj[0] == Oj[1]:
                dOij = 0
            elif (myObj[0] == Oj[0] and myObj[1] == Oj[1]) or (myObj[1] == Oj[0] and myObj[0] == Oj[1]):
                dOij = dis[idx2]

        dXij = abs(result[int(Oj[0] - 1)][iter] - result[int(Oj[1] - 1)][iter])
        newDis[idx1] = np.sqrt((dOij**2) - (dXij**2))

        if(newDis[idx1] >= maxDis):
            maxDis = newDis[idx1]
            maxObjA = Oj[0]
            maxObjB = Oj[1]

    maxObj.append(maxObjA)
    maxObj.append(maxObjB)
    maxObj.append(maxDis)
    return newDis

# Run k times to generate k project points of all the objects
def fastmap(Obj, dis, k):
    ObjA = 0
    ObjB = 0
    maxDis = 0
    iter = 0

    while iter < k:
        ObjA, ObjB, maxDis = findObj(Obj, dis, iter)
        project_xi(Obj, dis, ObjA, ObjB, maxDis, iter)
        iter = iter + 1
        if (iter < k):
            newDis = project_k_dimention(Obj, dis, iter - 1)
            dis = newDis

# Draw the result
def draw(label):
    for i in range(len(result)):
        plt.plot(result[i][0],result[i][1], 'c*', markersize = 5)
        plt.annotate(label[i], xy = (result[i][0],result[i][1]))
    plt.show()

def main():
    Obj, dis, label = readFile()
    List(Obj)
    fastmap(Obj, dis, k)
    print(result)
    draw(label)


if __name__ == "__main__":
    main()