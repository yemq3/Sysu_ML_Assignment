# coding=utf-8
'''
项目名称：
作者
日期
'''

# 导入必要库
import numpy as np

# 创建实验样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 代表侮辱性文字, 0 代表正常言论
    classVec = [0,1,0,1,0,1]    
    # postingList为词条切分后的文档集合，classVec为类别标签集合
    return postingList,classVec 


def createVocabList(dataSet):
    vocabSet = set([])
    for docment in dataSet:
        # 两个集合的并集
        vocabSet = vocabSet | set(docment) 
    # 转换成列表 
    return list(vocabSet) 

def setOfWords2Vec(vocabList,inputSet):
    # 创建一个与词汇表等长的向量，并将其元素都设置为0
    returnVec = [0]*len(vocabList)    
    for word in inputSet:
        if word in vocabList:
            #查找单词的索引
            returnVec[vocabList.index(word)] = 1 
        else: print ("the word: %s is not in my vocabulary" %word) 
    return returnVec


def train(trainMat,trainCategory):
    #trainMat:训练样本的词向量矩阵，每一行为一个邮件的词向量
    #trainGategory:对应的类别标签，值为0，1表示正常，垃圾
    numTrain = len(trainMat)
    numWords = len(trainMat[0])  
    pAbusive = sum(trainCategory)/float(numTrain)
    p0Num = np.ones(numWords); p1Num=np.ones(numWords)
    p0Denom=2.0; p1Denom=2.0
    for i in range(numTrain):
        if trainCategory[i] == 1:
            
            p1Num += trainMat[i] 
           
            p1Denom += sum(trainMat[i]) 
        else:
           
            p0Num += trainMat[i]
            
            p0Denom += sum(trainMat[i])
    # 类1中每个单词的概率
    p1Vec = p1Num/p1Denom 
    p0Vec = p0Num/p0Denom
    # 类0中每个单词的概率
    return p0Vec,p1Vec,pAbusive


def classfy(vec2classfy,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2classfy*p1Vec)+np.log(pClass1)
    p0 = sum(vec2classfy*p0Vec)+np.log(1-pClass1)
    if p1 > p0:
        return 1;
    else:
        return 0

# 对邮件的文本划分成词汇，长度小于2的默认为不是词汇，过滤掉即可。返回一串小写的拆分后的邮件信息。
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]


def bagOfWords2Vec(vocabList,inputSet):
    # vocablist为词汇表，inputSet为输入的邮件
    returnVec=[0]*len(vocabList)    
    for word in inputSet:
        if word in vocabList:
            #查找单词的索引
            returnVec[vocabList.index(word)] = 1 
        else: print ("the word is not in my vocabulary")
    return returnVec



def spamTest():
    fullTest = []
    docList = []
    classList= []
    # it only 25 doc in every class
    for i in range(1,26): 
        wordList = textParse(open('email/spam/%d.txt' % i,encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    # create vocabulary
    vocabList = createVocabList(docList)   
    trainSet = list(range(50));testSet=[]
    # choose 10 sample to test ,it index of trainMat
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainSet)))#num in 0-49
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat = []; trainClass = []
    for docIndex in trainSet:
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0, p1, pSpam = train(np.array(trainMat),np.array(trainClass))
    errCount = 0
    for docIndex in testSet:
        wordVec=bagOfWords2Vec(vocabList,docList[docIndex])
        if classfy(np.array(wordVec),p0,p1,pSpam) != classList[docIndex]:
            errCount += 1
            print (("classfication error"), docList[docIndex])

    print (("the error rate is ") , float(errCount)/len(testSet))

if __name__ == '__main__':
    spamTest()
