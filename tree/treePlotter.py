#coding=utf-8
from __future__ import print_function
import matplotlib.pyplot as plt
#全局变量
#dict() 函数用于创建一个字典 boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
#定义决策树的箭头属性
arrow_args=dict(arrowstyle="<-")

#绘制节点
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    # ax1 是函数 createPlot 的一个属性
    # annotate是关于一个数据点的文本  
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
        xytext=centerPt,textcoords='axes fraction',va="center",ha="center",\
        bbox=nodeType,arrowprops=arrow_args)

# def createPlot():
#     fig=plt.figure(1,facecolor="white")
      #把画布清空
#     fig.clf()
    #frameon是否显示坐标轴
#     createPlot.ax1=plt.subplot(111,frameon=False)
#     plotNode('juecejiedian',(0.5,0.1),(0.1,0.5),decisionNode)
#     plotNode('yejiedian',(0.8,0.1),(0.3,0.8),leafNode)
#     plt.show()
    
# 获得决策树的叶子结点数目
def getNumLeafs(myTree):
    #定义叶子节点数目
    numLeafs=0
    # 获得myTree的第一个键值，即第一个特征，分割的标签
    firstStr=myTree.keys()[0]
    #print("the firstStr is %s\n" %firstStr)
    # 根据键值得到对应的值，即根据第一个特征分类的结果
    secondDict=myTree[firstStr]
    #print("the secondDict is %s\n" %secondDict)
    # 遍历得到的secondDict
    for key in secondDict.keys():
         # 如果secondDict[key]为一个字典，即决策树结点
        if type(secondDict[key]).__name__=='dict':
            # 则递归的计算secondDict中的叶子结点数，并加到numLeafs上
            numLeafs+=getNumLeafs(secondDict[key])
        # 如果secondDict[key]为叶子结点 则将叶子节点+1
        else: numLeafs+=1
    return numLeafs


# 获得决策树的深度
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=myTree.keys()[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else: thisDepth=1
         # 如果当前树的深度比最大数的深度
        if thisDepth>maxDepth: maxDepth=thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
        {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTrees[i]

# 绘制中间文本 
def plotMidText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    # 绘制树结点
    createPlot.ax1.text(xMid,yMid,txtString)

# 绘制决策树
def plotTree(myTree,parentPt,nodeTxt):
    # 定义并获得决策树的叶子结点数
    numLeafs=getNumLeafs(myTree)
    #depth=getTreeDepth(myTree)
    # 得到第一个特征
    firstStr=myTree.keys()[0]
    # 计算坐标，x坐标为当前树的叶子结点数目除以整个树的叶子结点数再除以2，y为起点
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    # 绘制中间结点，即决策树结点，也是当前树的根结点
    plotMidText(cntrPt,parentPt,nodeTxt)
    # 绘制决策树结点
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    # 根据firstStr找到对应的值
    secondDict=myTree[firstStr]
    # 因为进入了下一层，所以y的坐标要变 ，图像坐标是从左上角为原点
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    # 遍历secondDict
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            # 计算叶子结点的横坐标 
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    # 计算纵坐标  
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD



def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    # 定义横纵坐标轴，无内容 
    axprops=dict(xticks=[],yticks=[])
    # 绘制图像，无边框，无坐标轴
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    # plotTree.totalW保存的是树的宽
    plotTree.totalW=float(getNumLeafs(inTree))
    # plotTree.totalD保存的是树的高
    plotTree.totalD=float(getTreeDepth(inTree))
    # 决策树起始横坐标和纵坐标
    plotTree.xOff=-0.5/plotTree.totalW;plotTree.yOff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


