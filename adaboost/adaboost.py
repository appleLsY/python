#coding=utf-8
from __future__ import print_function
from numpy import *

def loadSimpData():
    datMat=matrix([[1. , 2.1],[2. , 1.1],[1.3,1.],[1.,1.],[2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=ones((shape(dataMatrix)[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr);labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps=10.0;bestStump={};bestClasEst=mat(zeros((m,1)))
    minError=inf
    for i in range(n):
        rangeMin=dataMatrix[:,i].min();rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=mat(ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError=D.T*errArr
                #print('split: dim %d,thresh %.2f, thresh ineqal:\
                # %s,the weighted error is %.3f' %(i,threshVal,inequal,weightedError))
                if weightedError<minError:
                    minError=weightedError
                    bestClasEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst



#adaBoost算法
#@dataArr：数据矩阵
#@classLabels:标签向量
#@numIt:迭代次数   
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)
    aggClassEst=mat(zeros((m,1)))
    for i in range(numIt):
        #根据当前数据集，标签及权重建立最佳单层决策树
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        print('D:',D.T)
        #求单层决策树的系数alpha
        #max(error,1e-16)用于确保在没有错误时不会发生除零溢出
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
        #存储决策树的系数alpha到字典
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        print('classEst:',classEst.T)
        #预测正确为exp(-alpha),预测错误为exp(alpha)
        #即增大分类错误样本的权重，减少分类正确的数据点权重
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        #更新权值向量
        D=multiply(D,exp(expon))
        D=D/D.sum()
        #累加当前单层决策树的加权预测值
        aggClassEst+=alpha*classEst
        print('aggClassEst:',aggClassEst.T)
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate=aggErrors.sum()/m
        print('total error:',errorRate,'\n')
        if errorRate==0.0:break
    return weakClassArr