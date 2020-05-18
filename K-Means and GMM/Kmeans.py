# INF552 HW2
# Shih-Yu Lai 
# Shiuan-Chin Huang 
import random
import matplotlib.pyplot as plt
plt.style.use('ggplot')

data = [] # points from txt
centroid = [] # original centroids
cls = [] # the first class of all points
NewCen = []  # find new centroid
color = ['r', 'g', 'b', 'm', 'y', 'k']
MAX_ITER = 50

def readFile():
    with open('clusters.txt') as f:
    # with open('test.txt') as f:
        for line in f:
            if any(char.isdigit() for char in line) == True:
                line = line.replace("\n", "")
                line = line.split(',')
                data.append(line)

    for i in range(len(data)):
        plt.scatter(float(data[i][0]), float(data[i][1]))

    plt.show() # draw the original points

def RanCen(K): # random pich k centroids
    index = []
    First = random.randrange(len(data))
    centroid.append(data[First])
    index.append(First)

    for i in range(K - 1):
        New = True
        NewInd = random.randrange(len(data))
        for ind in range(index.__len__()): # check whether the index is already picked before
            if NewInd == index[ind]:
                New = False
            else:
                break

        if New:
            index.append(NewInd)
            centroid.append(data[NewInd])

def Centroid(K): # random pick the first centroid
    index = []
    # First = 9
    First = random.randrange(len(data))
    centroid.append(data[First])
    index.append(First)
    # print(centroid)
    for i in range(K - 1):
        Max = 0
        MaxInd = 0
        EDistence = []
        for j in range(len(data)):
            # pick the furthest point from the first centroid as the second one
            EDistence.append(Euclidean(data[j], centroid[i]))
            for ind in range(index.__len__()):
                if j == index[ind]:
                    break
                else:
                    temp = EDistence[j]*random.random()
                    if temp > Max:
                        Max = temp
                        MaxInd = j

        index.append(MaxInd)
        centroid.append(data[MaxInd])

    # for c in range(len(centroid)):
    #     plt.scatter(float(centroid[c][0]), float(centroid[c][1]), c = color[c])
    # plt.show()


def Euclidean(X, cen): # calculate the distance
    euclidean = 0
    for i in range(len(X)):
        euclidean += (float(X[i]) - float(cen[i])) ** 2
    return euclidean

def KMeans(K):
    # calculate the distance of all points to k centroids and classify them
    for i in range(len(data)):
        Min = 100000000
        cls.append(i)
        for j in range(len(centroid)):
            Dis = Euclidean(data[i], centroid[j])
            if Dis < Min:
                Min = Dis
                cls[i] = j
    # print(cls)

    Keep = True # while centrioid is still different, keep doing loop
    # CenDif = True
    time = 0

    # check if we need to keep finding the new centroid
    while time < MAX_ITER and Keep == True:
        Find_newCen(K)
        Find_newCls(K)
        Keep = check()
        time += 1
    # print(time)

def check():
    for cen in range(len(centroid)):
        if(NewCen[cen] != centroid[cen]):
            return True
    return False

def Find_newCls(K): # find new class based on new centroids
    NewCls = []
    for i in range(len(data)):
        Min = 100000000
        NewCls.append(i)
        for j in range(len(centroid)):
            Dis = Euclidean(data[i], NewCen[j])
            if Dis < Min:
                Min = Dis
                NewCls[i] = j
    # print(NewCls)

    Dif = False
    for ind in range(len(cls)):
        if(NewCls[ind] != cls[ind]):
            Dif = True
            break

    if Dif:
        for ind in range(len(cls)):
            cls[ind] = NewCls[ind]

def Find_newCen(K): # calculate new centroid
    for cen in range(len(NewCen)):
        centroid[cen] = NewCen[cen]
    NewCen.clear()
    for knum in range(K):
        X, Y, total = 0, 0, 0
        temp = []
        for i in range(len(data)):
            if (cls[i] == knum):
                total += 1
                X += float(data[i][0])
                Y += float(data[i][1])

        temp.append(str(X / total))
        temp.append(str(Y / total))
        NewCen.append(temp)

    # print(NewCen)

def Draw(K):
    for i in range(len(data)):
        for k in range(K):
            if(cls[i] == k):
                plt.scatter(float(data[i][0]), float(data[i][1]), c = color[k])

    for i in range(len(centroid)):
        plt.scatter(float(NewCen[i][0]), float(NewCen[i][1]), c = 'c')

    plt.show()

def main():
    readFile()
    K = 3
    Centroid(K)
    # RanCen(K)
    KMeans(K)
    Draw(K)

if __name__ == "__main__":
    main()