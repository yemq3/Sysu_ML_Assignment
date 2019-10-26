#-*-coding:utf-8-*-
import numpy as np

def loadDataSet(fileName):
    
    dataMat = []               
    fr = open(fileName)

    for line in fr.readlines():
        curLine = line.strip().split('\t')

        #将所有数据转换为float类型
        fltLine = list(map(float,curLine))  
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
          
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))     

def randCent(dataSet, k):
        
    #得到数据集的列数
    n = np.shape(dataSet)[1]          

    #得到一个K*N的空矩阵
    centroids = np.mat(np.zeros((k,n)))  

    #对于每一列
    for j in range(n):             

        #得到最小值
        minJ = min(dataSet[:,j])   

        #得到当前列的范围
        rangeJ = float(max(dataSet[:,j]) - minJ) 

        #在最小值和最大值之间取值
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1)) 
    return centroids

def distSLC(vecA, vecB):
    
    #pi为圆周率，在导入numpy时就会导入的了
    #sin(),cos()函数输出的是弧度为单位的数据
    #由于输入的经纬度是以角度为单位的，故要将其除以180再乘以pi转换为弧度
    #设所求点A ，纬度β1 ，经度α1 ；点B ，纬度β2 ，经度α2。则距离
    #距离 S=R·arc cos[cosβ1cosβ2cos（α1-α2）+sinβ1sinβ2]
    
    pi = np.pi

    a = np.sin(vecA[0,1]*pi/180) * np.sin(vecB[0,1]*pi/180)
    b = np.cos(vecA[0,1]*pi/180) * np.cos(vecB[0,1]*pi/180) * \
                      np.cos(pi * (vecB[0,0]-vecA[0,0]) /180)

    return np.arccos(a + b)*6371.0 #6371.0为地球半径

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
   
     
    # 数据集的行数，即数据的个数
    m = np.shape(dataSet)[0]             

    # 簇分配结果矩阵
    clusterAssment = np.mat(np.zeros((m,2)))

    # 第一列储存簇索引值
    # 第二列储存数据与对应质心的误差
    # 先随机生成k个随机质心的集合
    centroids = createCent(dataSet, k)
    clusterChanged = True

    # 当任意一个点的簇分配结果改变时
    while clusterChanged:             
        clusterChanged = False

        # 对数据集中的每一个数据
        for i in range(m):            
            minDist = np.inf; minIndex = -1

            # 对于每一质心
            for j in range(k):        

                # 得到数据与质心间的距离  
                distJI = distMeas(centroids[j,:],dataSet[i,:])

                # 更新最小值
                if distJI < minDist:  
                    minDist = distJI; minIndex = j

            # 若该点的簇分配结果改变
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2

        # print centroids
        # 对于每一个簇
        for cent in range(k):         

            # 通过数组过滤得到簇中所有数据
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]

            # .A 方法将matrix类型元素转化为array类型
            # 将质心更新为簇中所有数据的均值

            centroids[cent,:] = np.mean(ptsInClust, axis=0) 
            # axis=0表示沿矩阵的列方向计算均值
    return centroids, clusterAssment

import matplotlib
import matplotlib.pyplot as plt

def clusterPlaces(numClust=5):
    
    datList = []

    for line in open('Restaurant_Data_Beijing.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[0]),float(lineArr[1])])
    datMat = np.mat(datList)

    # 进行聚类
    myCentroids, clustAssing = kMeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()                                        

    # 创建一个矩形
    rect = [0.1,0.1,0.8,0.8]                                    

    # 用来标识簇的标记
    scatterMarkers = ['s' , 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']                  
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
   
    for i in range(numClust):                                 

        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)

    plt.show()
    
if __name__ == '__main__':
    clusterPlaces(6)
    
