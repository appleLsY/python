from numpy import *
import operator

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
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]