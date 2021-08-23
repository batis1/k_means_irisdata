# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:54:16 2021

@author: batis - 18511160002
"""

from numpy import *
import matplotlib.pyplot as plt

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCenttroid(dataSet, k):
    n = shape(dataSet)[1] # column
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):
        minJ = min(dataSet[:,j])  #1
        maxJ = max(dataSet[:,j]) #6
        rangeJ = float(maxJ - minJ) # 5
        centroids[:,j] = mat( minJ + rangeJ * random.rand(k,1) )
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCenttroid):
    m = shape(dataSet)[0]# all records in dataset
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):# test all records
            minDist = inf; minIndex = -1# clusterId
            for j in range(k):# test all centroids
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            # for record i, which centroid is closeset     
            clusterAssment[i,:] = minIndex,minDist**2
            #minIndex is index of closest centroid
            # minDist**2 is the closest distance
        #print(centroids) #
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
              # get all point  belong  to this cluster(center)
            centroids[cent,:] = mean(ptsInClust, axis=0)
              # get average  point from all the records belonged to this cluster as centroid
    return centroids, clusterAssment

def kMeansTest(indx,dataSet, k,clusterA,  c, distMeas=distEclud):
    clusterAssment = mat(zeros((1,2)))
    centroids = c
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        minDist = inf; minIndex = -1# clusterId
        for j in range(k):# test all centroids
            distJI = distMeas(centroids[j,:],indx)
            if distJI < minDist:
                minDist = distJI; minIndex = j
        if clusterAssment[0,0] != minIndex:
            clusterChanged = True
        # for record i, which centroid is closeset     
        clusterAssment[0,:] = minIndex,minDist**2
    return centroids, clusterAssment

def autoNorm(dataSet):# dataset 100 3
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))##(100,3)
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def split2(filename,sep=",",ratio=0.2, isRandome = False):
    fr = open(filename)
    numberOfLines = len(fr.readlines())-1
    fr.seek(0,0)
    line = fr.readline()
    colnum = len(line.split(sep))
    
    returnMat = zeros((numberOfLines,colnum-1))
    classLabelVercot = []
    
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFormLine = line.split(sep)
       
        if(len(listFormLine) > 1):
            returnMat[index,:] = listFormLine [0:colnum-1]
        classLabelVercot.append((listFormLine[-1]))
        index +=1
    classLabelVercot = array(classLabelVercot)
    normMat, ranges, minVals = autoNorm(returnMat)
    trainLines = numberOfLines - int(numberOfLines*ratio)
    
    if(isRandome):
        indexselect = CreateRandomList(numberOfLines,trainLines)
        Xtrain = normMat[indexselect]
        Ytrain = classLabelVercot[indexselect]
        Xtest = normMat[~indexselect]
        Ytest = classLabelVercot[~indexselect]
        return Xtrain, Ytrain,Xtest, Ytest
    
    
    Xtrain = normMat[0:trainLines,:]
    Ytrain = classLabelVercot[0:trainLines]
    Xtest = normMat[trainLines:,:]
    Ytest = classLabelVercot[trainLines:]
    
    return Xtrain, Ytrain,Xtest, Ytest

def CreateRandomList(maxNum,num):
    
    a=[False]*maxNum
    while sum(a)<num:
    
        inx=random.randint(0,maxNum-1)
        a[inx]=True
        
    return array(a)

def LabelValue(value):
    if value == "Iris-setosa":
        return 0.0
    elif value == "Iris-versicolor":
        return 1.0
    elif value == "Iris-virginica":
        return 2.0
    return -1

def TestIris():
    Xtrain,Ytrain,Xtest,Ytest=split2("irisdata-v2.txt",isRandome=True)
    len1=len(Xtest)
    print(len1)
    countCorrect= 0
    countError = 0
    
    centroids, clusterAssment = kMeans(Xtrain, 3)
    
    for i in range(len1):
        centroids, clusterAssment2 = kMeansTest(Xtest[i], Xtrain, 3,clusterAssment,centroids)
        print(clusterAssment2[0,0] )
        rows = shape(clusterAssment2)[0] # numbers of rows

        if clusterAssment2[0,0] == LabelValue(Ytest[i]):
            print("Correct\n")
            countCorrect=countCorrect+1
        else:
            print("Error\n")
            countError = countError+1

    print("Correct rate: ")
    print(countCorrect/len1)

    print("Erorr rate: ")
    print(countError/len1)
    

def clusterClubs(numClust=3):
    datList = []
    datMat,Ytrain,Xtest,Ytest=split2("irisdata-v2.txt",isRandome=True)
    datMat = mat(datList)
    myCentroids, clustAssing = kMeans(datMat, 3)
       #myCentroids=mat(myCentroids)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)    
    # the following statement is wrong I have to modify it
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
    
    
if __name__ == '__main__':
     TestIris()
     # clusterClubs()
