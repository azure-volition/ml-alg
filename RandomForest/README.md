# Random Forest

## 相关知识

* 决策树: [Decision Trees Tutorial by Andrew Moore](http://www.autonlab.org/tutorials/dtree.html)
* 信息增益: [Information Gain Tutorial by Andrew More](http://www.autonlab.org/tutorials/infogain.html)
* [机器学习实践](http://www.ituring.com.cn/book/1021) 第3章 决策树

## 简介及例子

* [博客 random-forest-and-gbdt（快速了解）](http://www.cnblogs.com/LeftNotEasy/archive/2011/03/07/random-forest-and-gbdt.html)
* [Breiman and Cutler的介绍及代码示例（快速了解）](../ml-pdf/randomforest.pdf)
* [using random forest(milk库)](https://pythonhosted.org/milk/randomforests.html)

## 深入介绍
* [Breiman and Cutler的详细教材](http://www.stat.berkeley.edu/~breiman/RandomForests/)
* [微软的Paper](http://research.microsoft.com/apps/pubs/default.aspx?id=155552)

## 优点

* 表现良好，在当前的很多数据集上，相对其他算法有着很大的优势
* 能够处理很高维度（feature很多）的数据，并且不用做特征选择
* 训练完后，它能够给出哪些feature比较重要
* 创建随机森林的时候，对generlization error使用的是无偏估计
* 训练速度快
* 在训练过程中，能够检测到feature间的互相影响
* 容易做成并行化方法，实现比较简单

## 算法（分类 & 回归）

* 组成很多棵决策树（或回归树）组成，决策树（或回归树）之间没有关联
* 新样本预测：将新样本输入给所有决策树分类（或回归），将投票结果（或均值）作为最终的分类（或回归）结果
* 模型训练：
	* 参数：树的数量，每棵树选择多少个特征 
	* 采样，在不进行剪枝的情况下避免 overfitting
	
		~~~
		Bootstrap Samples: 为Ntree棵树建立Ntree个样本集，每个样本集样本数为M（从M个总样本中随机抽取M次，会有样本被漏选、也会有样本被重复选入）
		Randomly Sample: 每棵树从N个特征中随机选取X个特征进行训练(X<<N)
		~~~
	
	* 用完全分裂的方式建决策树：
		
		~~~
		Best Split：叶子节点要么无法继续分裂、要么所有样本都指向同一个类别
		~~~
		
	* 训练错误率估算：
	
		~~~
		OOB(Out Of Bagging)：建立Bootstrap Sample时，没有被选入的样本
		~~~
	
		* 在每轮迭代（训练一棵树）时，没有被随机抽取到的样本，被用来估算这轮迭代的训练错误率
		* 训练完随机森林后，用所有的OOB样本来估算整个模型的训练错误率
		
* 获取的额外信息：
	* 特征重要程度：根据各棵随机树OOB样本的数量来估算哪些特征对分类结果影响最大
	* Matrix of Proximity Measures among the input：based on the frequency that pairs of data points are in the same terminal nodes

* 分类
	* 单棵树选取特征数：总特征数^(1/2) 
	* Default Node Size = 1
	* 特征重要程度有4种度量

* 回归
	* 单棵树选取特征数：总特征数 / 3
	* Default Node Size = 5：在构建决策树时，节点数小于这个值将不再分裂
	* 特征重要程度有1种度量

* 非监督学习
	* 使用模拟构造样本分类结果，见[参考文件](../ml-pdf/randomforest.pdf)

## 库

[sklearn.ensemble.RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

[sklearn.ensemble.RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)

[implementation列表](http://butleranalytics.com/random-forest-implementations/)


