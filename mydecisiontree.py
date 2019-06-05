import operator
from math import log


## 1.加载数据
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels
dataSet,labels = createDataSet()



##计算系统熵函数
#这个函数的目的就是求系统熵,其本质就是求传入的数据集里面所对应的"系统"熵
def calcshannonEnt(dataSet):
    numEntries = len(dataSet)
    # print(numEntries)
    labelCounts = {}
    for featVec in dataSet:
        # print("----循环开始了-----")
        # print(featVec)
        # [1, 1, 'yes']
        # [1, 1, 'yes']
        # [1, 0, 'no']
        # [0, 1, 'no']
        # [0, 1, 'no']
        currentLabel = featVec[-1]
        # print('----------每次循环都有我----------')
        # print(currentLabel)                                                     ##yes---》yes----->no------->no-------->no
        if currentLabel not in labelCounts.keys():labelCounts[currentLabel] = 0 ##如果currentLabel不在labelCounts里面，labelCounts[currentLabel]就为0
                                                                                ##就是把yes和no放到currentLabel这个字典里面
        # print(labelCounts)
        labelCounts[currentLabel] += 1                                          #这里的这个技巧同Knn,利用字典进行计数
        # print(labelCounts)
        # labelCounts[currentLabel]=labelCounts.get(currentLabel,0) +1          #等价于上上面一行
    # print(labelCounts)                                                           #{'yes': 2, 'no': 3}
    shannonEnt = 0.0                                                               #熵一开始为0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries                               #如果是yes,这里是2/5,如果是no，则这里是3/5
        # print(prob)
        shannonEnt -= prob * log(prob,2)                                        #求系统熵
        # print(shannonEnt)
    return shannonEnt

## 划分数据集
## 这里的axis是特征(这里的例子只有两个特征(不浮出水面会不会挂 和 有没有鱼鳍 ),用0和1表示)；
# value表示某一个特征所对应的具体的特征值
## 比如说:不浮出水面会不会挂,五个样本对应的值为1，1，1，0，0,这里的1和0就是value
##这个函数的作用:将某一个特征所对应的不同特征值划分出来,并且将特征值所对应的结果输出
##比如这里:特征0所对应的五个值为1,1,1,0,0,这里将三个1和2个0划分出来了( 通过if featVec[axis] ==value:这句话)
##但是这里需要注意的是:这里的3个1所对应的结果为yes,yes,no(这里针对的是这个例子),
##这里的这个函数输出的是yes,yes,no,no之类的,yes,no前面的1,1,0之类的不用看
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        # print(featVec)          #打印结果:[1, 1, 'yes'] [1, 1, 'yes'] [1, 0, 'no']...
        if featVec[axis] ==value:
            reducedFeattVec = featVec[:axis]
            reducedFeattVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeattVec)
            # print(retDataSet)
            ################################
            ####list列表的一些用法
            # a = [1, 2, 'yes']
            # print(a[:0])  # []
            # print(a[:1])  # [1]
            # print(a[:2])  # [1, 2]
            # print(a[:3])  # [1, 2, 'yes']
            # print(a[1:])  # [2, 'yes']
            # print(a[0:])  # [1, 2, 'yes']
            # print(a[2:])  # ['yes']
            ####extend和append的区别
            # b = [1, 3, 6]，a = [1, 2, 'yes']
            # a.extend(b)
            # print(a)  # [1, 2, 'yes', 1, 3, 6]
            # c = [1, 9]
            # a.append(c)
            # print(a)  # [1, 2, 'yes', 1, 3, 6, [1, 9]]
            #################################
    return retDataSet
# print(splitDataSet(dataSet,0,0))#[[1, 'no'], [1, 'no']]
# print(splitDataSet(dataSet,1,0))#[[1, 'no']]
# print(splitDataSet(dataSet,1,1))#[[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']],这里主要是看yes和no,yes和no前面的1,0不要看
# 上面这一行主要输出的东西主要看splitDataSet(dataSet,1,1)中的1，1;第一个1指的是特征,第二个1指的是value,可以发现有两个yes,两个no

## 求最优特征(这里有用到计算系统熵,划分数据集函数)
## 这个函数的主要思路就是计算出条件熵而后再计算出信息增益从而获得最优特征
##函数作用:传进数据集之后,返回一个最好的特征
def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataSet[0]) - 1         #特征数,这个例子里面的特征数为2个
    # print(len(dataSet[0]))                  #3，这里的dataSet[0]指的是[1, 1, 'yes']的长度
    # print(len(dataSet))                     #5
    # print(len(dataSet[4]))                  #3
    baseEntropy = calcshannonEnt(dataSet)     #这里是得到系统熵
    # print(baseEntropy)
    bestinfoGain = 0.0                        #初始化信息增益
    bestFeature = -1                          #初始化最好的特征
    for i in  range(numFeatures):
        # print(i)
        featList = [example[i] for example in dataSet]      #取出特征里面的特征值
        # print(featList)                                   #特征0-----》这里是对应的特征值[1, 1, 1, 0, 0]；1----》[1, 1, 0, 1, 1]
        # print([example for example in dataSet])           #[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
        #防止重复
        uniqueVals = set(featList)
        # print(uniqueVals)                                  #{0, 1}
        newEntropy = 0.0
        for value in uniqueVals:                             #这里只会循环2次
            subDataSet = splitDataSet(dataSet,i,value)
            print(subDataSet)
            prob = len(subDataSet) / float(len(dataSet))     #这里获取的是计算条件熵时候所用到的比例
                                                             #这里所特征1所对应4个是和一个否,而4个是里面对应两个yes两个否
            newEntropy += prob * calcshannonEnt(subDataSet)  #这里求的是条件熵
        infoGain = baseEntropy - newEntropy                  #这里求的是信息增益
        if (infoGain > bestinfoGain):
            bestinfoGain =infoGain
            bestFeature = i                                  #这里的特征就两个，经过比较(这里是通过for循环实现的),这里最后比较好的是特征是0，即:不浮出表面是否能够生存
    return bestFeature

##字典排序(knn那边有)
##这个函数的作用:将传入的列表里面的元素进行统计,放到一个字典里面,统计出每一个元素的数量,
##而后转换为数组(数组里面是按照降序排列),最后返回出现次数最多的那个元素
## 主要作用就是返回传入的列表里面数量最多的元素
def majorityCnt(classList):
    classConut = {}
    for vote in classList:
        if vote not in classConut.keys():classConut[vote] = 0
        classConut[vote] +=1
        # classConut[vote] =classConut.get(vote,0) +1         #这一行等价于上一行
    sortedClassCount = sorted(classConut.items(),key=operator.itemgetter(1),reverse=True)
    ###items 的例子: https://www.runoob.com/python/att-dictionary-items.html
    # dict = {'Google': 'www.google.com', 'Runoob': 'www.runoob.com', 'taobao': 'www.taobao.com'}
    # # print("字典值:%s" %dict.items())
    # print(dict.items())  # dict_items([('Google', 'www.google.com'), ('Runoob', 'www.runoob.com'), ('taobao', 'www.taobao.com')])
    return sortedClassCount[0][0]



##创建决策树(这里有用到划分数据集函数,求最优特征函数(这里有用到计算系统熵,划分数据集函数))
##
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # classList = [example[-1] for example in dataSet]

    # print(classList)                                                                                      #['yes', 'yes', 'no', 'no', 'no']
    if classList.count(classList[0])==len(classList):                                                       #当类别相同的时候直接返回所有类别标签；这里为[0,0] 的时候，对应类别为no,no,这里可以直接返回
    # if classList.count(classList[0]) == len(classList):
    #     print(classList[0])                                                                               #当数据集对应的类别相同时,返回该类别
        return classList[0]
    #count的用法：返回元素在列表中出现的次数。
    # aList = [123, 'xyz', 'zara', 'abc', 123];
    # print(aList.count(123))  # 2
    # print(aList.count('xyz'))  # 1
    # print(aList.count('zara'))  # 1
    if len(dataSet[0]) == 1:                                                                                #使用完所有特征仍然不能把数据集划分成包含唯一类别的分组,则返回出现次数最多的类别作为返回值
        return majorityCnt(classList)
    besfFeat = chooseBestFeatureToSplit(dataSet)
    # print(besfFeat)                                                                                       #这里为0
    besfFeatLabel = labels[besfFeat]
    # print(besfFeatLabel)                                                                                   #no surfacing
    myTree = {besfFeatLabel:{}}                                                                              #这里树的结构是以字典表现出来的
    # print(myTree)
    del (labels[besfFeat])
    featValues = [example[besfFeat] for example in dataSet]
    # print(featValues)                                                 #[1, 1, 1, 0, 0]
    uniqueVals = set(featValues)

    for value in  uniqueVals:
        subLabels = labels[:]                                                                            #复制标签
        # print("--------------")
        # print(value)                                                                                    #第一次为0，第二次为1
        # print(subLabels)                                                                                #[]
        myTree[besfFeatLabel][value] = createTree(splitDataSet(dataSet,besfFeat,value),subLabels)         #这里利用递推的技巧
        # print(myTree[besfFeatLabel][value])                                                             #{'flippers': {0: 'no', 1: 'yes'}}
        # print(myTree[besfFeatLabel])                                                                    #{0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
        # print(myTree)                                                                                   #{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        ##myTree[besfFeatLabel][value]的有关解释:
        # dic = {"hello": {"world": 2}}
        # print(dic["hello"]["world"])  # 2
        # dic = {"hello": {"world": 1}}
        # dic["hello"]["world"] = 3
        # print(dic)  # {'hello': {'world': 3}}
    print("----------------分界线-----------")
    print(myTree)
    return myTree

##注意:
##createTree函数作用:
# 1.把数据集传到createTree这个函数
##2.把特征对应的不同的特征值所对应的数据集交给它，就可一个通过字典的方式把整个数取出来
##














# calcshannonEnt(dataSet)
# print(splitDataSet(dataSet,0,0))#[[1, 'no'], [1, 'no']]
# print(splitDataSet(dataSet,1,0))#[[1, 'no']]
# print(splitDataSet(dataSet,1,1))#[[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']]
# chooseBestFeatureToSplit(dataSet)

createTree(dataSet,labels)
