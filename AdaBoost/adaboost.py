# coding: UTF-8
'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():
    # 样本特征：X1, X2
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    # 样本分类结果：Y
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):
    # 从'\t'分隔文件中加载样本，最后一列是分类结果，其他列为特征
    numFeat  = len(open(fileName).readline().split('\t')) 
    dataMat  = []
    labelMat = []
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
    # 简化版单层决策树（根据阈值简单分类，只分一层，没有引入信息增益等概念）
    # dataMatrix中，下标dimen所在元素被不满足条件命中(threshIneq thresVal)的样本（行），被分类为 -1.0
    retArray = ones((shape(dataMatrix)[0],1)) 
    if threshIneq == 'lt':
        # dataMatix[:,dimen] <= threshVal 返回bool向量，例如
        #    >>> m
        #    matrix([[1, 2],  [3, 4],  [5, 6]])
        #    >>> m[:,0] <= 3.0
        #    matrix([[True],  [True], [False]], dtype=bool)
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] >  threshVal] = -1.0
    # 返回 m*1 矩阵， 对m个样本进行分类
    return retArray
    

def buildStump(dataArr,classLabels,D):
    # 功能：构建一个弱分类器（本例是一个简化版的决策树模型）
    # 参数：dataArr(m*n)m个样本的特征；classLabels(m*1)标注的分类结果；D（m*1）m个样本的权重、训练之前各元素初始化为1/m
    # 流程：遍历stumpClassify所有可能的输入值，找到数据集上“最佳”（基于数据的权重向量D来定义）的单层决策树
    # 样本
    dataMatrix  = mat(dataArr);       # m*n 矩阵：m个样本，n个特征
    labelMat    = mat(classLabels).T  # 1*m 矩阵：转置后，m*1矩阵变为1*m矩阵
    m,n         = shape(dataMatrix)   # 
    # 变量
    numSteps    = 10.0;               # 在每个特征的取值范围内，平均打numSteps个点，以用来覆盖特征的各种分类阈值
    bestStump   = {};                 # 给定权重向量D时，最佳决策树的相关信息
    bestClasEst = mat(zeros((m,1)))   # m*1 矩阵 (向量)
    minError    = inf                 # 初始化为正无穷大，以便后续找到最小的错误率
    for i in range(n): 
    # 遍历所有的特征（维度）
        rangeMin = dataMatrix[:,i].min()        # 特征i的最小值
        rangeMax = dataMatrix[:,i].max()        # 特征i的最大值
        stepSize = (rangeMax-rangeMin)/numSteps # 步长
        for j in range(-1,int(numSteps)+1):
        # 遍历当前特征的所有取值：rangMin + float(j)*stepSize
            for inequal in ['lt', 'gt']: 
            # 在“大于”，“小于”两种不等式之间切换
                threshVal                       = (rangeMin + float(j) * stepSize)              # 分类阈值
                predictedVals                   = stumpClassify(dataMatrix,i,threshVal,inequal) # 在样本 dataMatrix 上，根据特征i，以threashVal为阈值，inequal为1/-1判别标准进行分类
                errArr                          = mat(ones((m,1)))                              # m*1矩阵（向量）中，分错的样本被保留为1
                errArr[predictedVals==labelMat] = 0                                             # m*1矩阵（向量）中，分对的样本被设置为0
                weightedError                   = D.T*errArr                                    # 这一个弱分类器的错误权重（分错样本的权值之和）
                # 调试日志： print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    # 找到了错误率更低的模型参数(决策树所选特征，分类阈值，大于阈值时分为1还是小于时分为1)
                    # 更新最佳决策树信息
                    minError            = weightedError         # 错误权重
                    bestClasEst         = predictedVals.copy()  # 样本分类结果
                    bestStump['dim']    = i                     # 根据哪个特征分类
                    bestStump['thresh'] = threshVal             # 分类阈值
                    bestStump['ineq']   = inequal               # 大于阈值时分为1，还是小于等于时分为1
    # 返回：决策树参数，错误权重，分类结果
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    # 功能：训练adaBoost分类器（DS是单层决策树Decision Stump缩写，会训练出一组DS）
    # 输入：样本特征(m*n矩阵，m个样本)，样本类别（1*m矩阵，转置后变成m*1矩阵，即向量），最大迭代次数
    # 输出：单层决策树数组，各样本累积的分类结果
    weakClassArr = []                   # 弱分类器数组
    m            = shape(dataArr)[0]    # m个样本（dataArr是m*1矩阵）
    D            = mat(ones((m,1))/m)   # m个样本的权重，初始化为全部相等(1/m)；之后的迭代中，分错的样本权重会被增加，分对的样本权重会被减少
    aggClassEst  = mat(zeros((m,1)))    # m个样本的类别估计累计值？？？
    for i in range(numIt):
    # 最多运行numIt轮，错误率为0时提前结束
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        # 用样本（dataArr,classLabels）及样本权重向量（D），构建单层分类树bestStump，得到错误权重error（由D和分错样本计算），m个样本分类结果classEst
        # 调试日志：print "D:",D.T
        # alpha：本次单层决策树输出结果的权重, max(error,le-16)为了防止error=0时除零异常
        # alpha计算公式为：0.5*log((1-error_pct)/error_pct)
        alpha              = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha  
        # 将训练出来的弱分类器追加到数组中
        weakClassArr.append(bestStump)
        # 调试日志：print "classEst: ",classEst.T
        # 更新样本参数D，更新公式：
        #   样本分类正确，则 D<t+1,i> = D<t,i> * e^(-alpha) / sum(D<t,*>)
        #   样本分类错误，则 D<t+1,i> = D<t,i> * e^(+alpha) / sum(D<t,*>)
        # multiply函数用于向量点积，先用mat(classLabels).T将(1*m)矩阵转置为向量(m*1)
        # 计算点积时，分对的样本 1*1, -1*-1都为1，分错的样本 1*-1, -1*1为-1
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D     = multiply(D,exp(expon))
        D     = D/D.sum()
        # 调试日志：calc training error of all classifiers, if this is 0 quit for loop early (use break)
        # 更新累积分类结果：+ 分类器权重 * 分类结果；aggClassEst, classEst都是m*1向量
        aggClassEst += alpha*classEst
        # 调试日志：print "aggClassEst: ",aggClassEst.T
        # 错误率（分错的样本数/样本总数）为0时提前结束，sign函数输入mat[[1.2],[-0.6],[2.1]]输出mat[[1],[-1],[1]]
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T, ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        if errorRate == 0.0: break
    # 返回：弱分类器数组
    #      各样本分类估计累计值(sum(alpha<i>*classEst<i>))，其中alapha<i>为第i个弱分类器的权重，classEst<i>为第i个弱分类器对该样本的分类结果(1或-1)
    #      (返回aggClassEst，能够得到置信强度，以便绘制POC曲线)
    # 本质上还是弱分类器加权，但是：
    #    弱分类器训练时更关注先前被分错的样本（借助D表示）
    #    在更新D时，准确程度（alapha）越高的弱分类器，（分对或分错）时对D的影响能力越大
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    # 功能：对新样本的分类进行预测
    # 参数：m个daifenlei 
    dataMatrix  = mat(datToClass)       # m个待分类样本
    m           = shape(dataMatrix)[0]  # 样本数
    aggClassEst = mat(zeros((m,1)))     # m*1矩阵（向量），存储m个样本的分类结果
    for i in range(len(classifierArr)): 
    # 用每个弱分类器分类，权重是弱分类器的 alpha 值
        # classEst 是 m*1 矩阵
        classEst    =  stumpClassify( \
                                 dataMatrix, \
                                 classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq']) 
        aggClassEst += classifierArr[i]['alpha']*classEst
        # 打印日志，可以看出，随着迭代进行，aggClassEst的值越来越强
        print aggClassEst
    # 返回分类结果（m*1矩阵，元素值为-1或1）
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    # 功能：为模型的测试结果，绘制ROC曲线（x轴是假阳率，y轴是真阳率），来评估正确率、召回率之间的trade off
    # 输入：predStrengths 是各个样本的分类预测强度（是numpy.Array或1*m矩阵（向量转置））
    #      classLabels   是各个样本的真实类别（是长度为m的普通列表）
    # predStrengths来源：可以是朴素贝叶斯的置信概率，可以是Logistic回归输入到Sigmoid函数的数值，可以是AdaBoost/SVM输入给sign函数的数值
    import matplotlib.pyplot as plt
    cur            = (1.0,1.0)                              # 绘制光标的位置
    ySum           = 0.0                                    # 计算出来的AUC（Area Unser the Curve，ROC曲线下面积）的值
    numPosClas     = sum(array(classLabels)==1.0)           # 真实正例的数目
    yStep          = 1/float(numPosClas)                    # y轴(真阳率)的遍历步长：1/正例的数目
    xStep          = 1/float(len(classLabels)-numPosClas)   # x轴(假阳率)的遍历步长：1/负例的数目
    sortedIndicies = predStrengths.argsort()                # 按置信强度升序排列时的数组下标
    fig            = plt.figure()                           # 
    fig.clf()                                               # 清理一下figure对象
    ax             = plt.subplot(111)                       # subplot(nrows, ncols, plot_number)
    for index in sortedIndicies.tolist()[0]:
    # 按置信度由小到大遍历样本（从右上向左下绘制曲线）
        if classLabels[index] == 1.0:
        # 样本的真实分类是正例：消耗了1个正例，y轴（表示真阳率）坐标在这里要降低一个单位
            delX =  0
            delY =  yStep
        else:
        # 样本的真是分类是负例：消耗了1个负例，x轴（表示假阳率）坐标在这里要降低一个单位
            delX =  xStep
            delY =  0
            ySum += cur[1]  # x轴坐标每变化一次，都累加一下当前的y值到AUC（曲线下面积）
        # 绘制一段曲线：从坐标cur到(cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        # 更新坐标cur
        cur = (cur[0]-delX,cur[1]-delY)
    # 绘制图像其他部分
    ax.plot([0,1],[0,1],'b--') # 绘制从(0,0)到(1,1)的对角线虚线，给出随机猜测的结果曲线
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate') 
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])     
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep
