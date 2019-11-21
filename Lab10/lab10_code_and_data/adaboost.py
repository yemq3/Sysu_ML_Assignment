# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:28:48 2019

@author: user3
"""

import numpy as np

def loadSimpData():
    datMat = np.matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):     
    '''
    输入：数据以tab键分隔的txt文件
    输出：格式化的数据集，分类标签列表
    描述：读取文件数据，格式化为算法可处理的格式
    '''
    numFeat = len(open(fileName).readline().split('\t')) 
    dataMat = []; labelMat = []
    fr = open(fileName)

    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t') 
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    '''
    输入：数据集，属性对应的列下标，阈值，判断大于阈值还是小于阈值的值被分类到-1的变量
    输出：对每个样本预测分类的数组
    描述：通过阈值比较对数据分类，在阈值两边的数据分别会被分到类别+1和-1中
    '''
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    
    if threshIneq == 'lt':                   
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    elif threshIneq == 'gt':              
        retArray[dataMatrix[:,dimen]  > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D):
    '''
    输入：样本数据，类标签列表，样本数据权重向量
    输出：字典形式的最佳单层决策树，最小误差率，储存类别估计值的列向量
    描述：遍历stumpClassify()函数所有可能输入值，找到基于权重向量D的最佳单层决策树
    '''
    dataMatrix = np.mat(dataArr);     
    labelMat = np.mat(classLabels).T 
    m,n = np.shape(dataMatrix)        
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf                 

    for i in range(n):             
        rangeMin = dataMatrix[:,i].min();      
        rangeMax = dataMatrix[:,i].max();      
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):   

            for inequal in ['lt', 'gt']:       
                threshVal = (rangeMin + float(j) * stepSize)                 
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))      
                
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr    
               
                #print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" \
                #% (i, threshVal, inequal, weightedError))

                if weightedError < minError:   
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    '''
    输入：样本数据，类标签列表，弱分类器迭代次数
    输出：包含每一次迭代得到的弱分类器的数组，记录每个数据点的类别估计累计值的列向量（在迭代过程中不断优化）
    描述：基于单层决策树的adaBoost训练过程
    '''
    weakClassArr = []
    m = np.shape(dataArr)[0]    
    D = np.mat(np.ones((m,1))/m)   
    aggClassEst = np.mat(np.zeros((m,1)))

    for i in range(numIt):   #迭代numIt次
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)

        #print("D:",D.T)

        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha                          
        weakClassArr.append(bestStump)                      

        #print("classEst: ",classEst.T)
        
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst) 
        D = np.multiply(D,np.exp(expon))                             
        D = D/D.sum()                                          
        
        aggClassEst += alpha*classEst

        #print("aggClassEst: ",aggClassEst.T)
        
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m

        #print("total error: ",errorRate)

        if errorRate == 0.0: break 
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    '''
    输入：样本数据，弱分类器列表
    输出：类别估计累计值的列向量
    描述：基于adaBoost进行分类
    '''
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0] 
    aggClassEst = np.mat(np.zeros((m,1)))

    for i in range(len(classifierArr[0])):
        
        classEst = stumpClassify(dataMatrix,classifierArr[0][i]['dim'],\
                                 classifierArr[0][i]['thresh'],\
                                 classifierArr[0][i]['ineq'])
        aggClassEst += classifierArr[0][i]['alpha']*classEst
        #print(aggClassEst)
    return np.sign(aggClassEst)

if __name__ == "__main__":
    dataArr,labelArr = loadDataSet('horseColicTraining2.txt')  
    testArr,testLabelArr = loadDataSet('horseColicTest2.txt') 
    numIt_list=[1,10,50,100,300,500,750,1000,1500]            

    for i in range(size(numIt_list)):
        print("\n%d.When numIt is %d:\n" % (i,numIt_list[i]))
        classifierArray,aggClassEst = adaBoostTrainDS(dataArr,labelArr,numIt_list[i])
        #以下3行代码求训练误差率
        m = np.shape(np.mat(dataArr))[0]
        errArr = np.mat(np.ones((m,1)))
        TrainingErrorRate = errArr[np.sign(aggClassEst) != np.mat(labelArr).T].sum()
        print("->The error rate in training process is %f %%\n" % TrainingErrorRate)
        #以下3行代码求测试误差率
        prediction = adaClassify(testArr,classifierArray)
        errArr = np.mat(np.ones((67,1)))
        TestingErrorRate = errArr[prediction!=mat(testLabelArr).T].sum()
        print("->The error rate in testing process is %f %%\n" % TestingErrorRate)
