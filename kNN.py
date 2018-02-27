from numpy import *
import operator

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify(inX,dataSet,labels,k): #inX是用于分类的输入向量 dataSet是训练样本集 labels是标签向量 k表示用于选择最近邻居的数目
    dataSetSize=dataSet.shape[0]  
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiddMat=diffMat**2
    sqDistances=sqDiddMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]