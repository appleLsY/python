#coding=utf-8
from __future__ import print_function
from numpy import *
import operator
import os

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

#inX是用于分类的输入向量 dataSet是训练样本集 labels是标签向量 k表示用于选择最近邻居的数目
def classify(inX,dataSet,labels,k): 
    #shape是用来获取数组的长度，shape【0】则是获取第一维的数据。这里为4，因为有4行
    dataSetSize=dataSet.shape[0]  
    #tile（A，reps）其中A是原始数组，reps则是要进行重复的次数
    #tile(a,(x,y))：   结果是一个二维矩阵，其中行数为x，列数是一维数组a的长度和y的乘积
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    #平方
    sqDiddMat=diffMat**2
    #不指定axis参数时是对数组中的所有元素求总和
    #指定axis参数时可以按行/按列求和，求和的结果相比于原数组降低了一个维度
    sqDistances=sqDiddMat.sum(axis=1)
    #开根号
    distances=sqDistances**0.5
    #argsort函数返回的是数组值从小到大的索引值
    sortedDistIndicies=distances.argsort()
    classCount={}
    #range（a，b）就是从a到b
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        #get（k，d）相当于一条if…else…语句。若k在字典a中，则返回a[k]；若k不在a中，则返回参数d
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #sorted是内置排序函数
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#从文本文件中解析数据
def file2matrix(filename):
    fr=open(filename)
    #文件中的总列数
    arrayOLines=fr.readlines()
    #len 代表是元素个数
    numberOfLines=len(arrayOLines)
    #zeros是返回一个double类零矩阵，其中有numberOfLines行，3列
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        #strip() 方法用于移除字符串头尾指定的字符（默认为空格）
        line=line.strip()
        #split()通过指定分隔符对字符串进行切片，如果参数num 有指定值，则仅分隔 num 个子字符串
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        #append() 方法用于在列表末尾添加新的对象
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#测试代码
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCoun=0.0
    for i in range(numTestVecs):
        classifierResult=classify(normMat[i,:],normMat[numTestVecs:m,:],\
            datingLabels[numTestVecs:m],3)
        print("the classifier came back with %d,the real answer is: %d" %(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errorCoun+=1.0
    print("the errorid %d, the total error rate is :%f" %(errorCoun,errorCoun/float(numTestVecs)))

#预测代码
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(input("percentage of time spent playing video games?"))
    ffMiles=float(input("frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream comsumede per year?"))
    datingDataMat,datingLabels=file2matrix('datingTest.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will probably like this person:",resultList[classifierResult-1])

#手写识别系统
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels=[]
    #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    # 这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中
    trainingFileList=os.listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s' %fileNameStr)
    testFileList=os.listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s' %fileNameStr)
        classifierResult=classify(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d,the real answer is: %d"\
         %(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errorCount+=1.0
    print("\nthe total number of errors is: %d" %errorCount)
    print("\nthe total error rate is: %f" %(errorCount/float(mTest)))