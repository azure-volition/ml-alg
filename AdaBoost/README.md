## AdaBoosting代码注释

> [用AdaBoost将单层决策树(decision stummp)集成为一个强分类器](https://github.com/fangkun/ml_notes_codes/blob/master/AdaBoost/adaboost.py)

## AdaBoosting概要

### 所属类别

1. meta-algorithm, ensemble-method
2. bagging: [Random Foresst](http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)
3. boosting: AdaBoost

### 原理及实现

1. 一般流程

2. 算法原理
	
	> 	串行训练N个弱分类器
	
	> 	训练时：每个弱分类器关注之前分错的样本，通过M个样本的权值向量D（M维）来实现
	
	> 	预测时：所有弱分类器预测结果加权求和，作为最终预测结果，权值是弱分类器的准确度alpha

	>  ![](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101022146359950.png)
	
3. 权值向量D
	
	> 	初始：M*1矩阵，所有样本权重相等，都是1/M
	
	> 	更新：在训练第t+1个弱分类器时，对于D中第i个元素(第i个样本的权值）
	
	>  	分对时，降低这个样本的权值
	
	> 	<math display="block">
	> 	<msubsup><mi>D</mi><mi>i</mi><mi>|t+1|</mi></msubsup>
	> 	<mo>=</mo>
	> 	<mo>(</mo>
	> 	<msubsup><mi>D</mi><mi>i</mi><mi>|t|</mi></msubsup>
	> 	<mo>*</mo>
	> 	<msup><mi>e</mi><mi>^-&alpha;</mi></msup>
	> 	<mo>)</mo>
	> 	<mo>/</mo>
	> 	<mi>Sum(D)</mi>
	> 	</math>
	
	>  分错时，提升这个样本的权值，在训练下一个弱分类器时更被“重视”
	 
	> 	<math display="block">
	> 	<msubsup><mi>D</mi><mi>i</mi><mi>|t+1|</mi></msubsup>
	> 	<mo>=</mo>
	> 	<mo>(</mo>
	> 	<msubsup><mi>D</mi><mi>i</mi><mi>|t|</mi></msubsup>
	> 	<mo>*</mo>
	> 	<msup><mi>e</mi><mi>^ &alpha;</mi></msup>
	> 	<mo>)</mo>
	> 	<mo>/</mo>
	> 	<mi>Sum(D)</mi>
	>	</math>
	
	>	D在每一轮中都被更新，直到所有弱分类器都被训练完毕
	
	>	其中alpha（弱分类器的置信强度）越高，对D的影响越大
	
4. 弱分类器置信强度alpha
	
	> 	<math display="block">
	> 	<mi>&alpha;</mi>
	> 	<mo>=</mo>
	> 	<mi>0.5</mi>
	> 	<mo>ln</mo>
	> 	<mo>[</mo>
	> 	<mo>(</mo> <mi>1</mi> <mo>-</mo> <mi>&epsilon;</mi> <mo>)</mo>
	> 	<mo>/</mo>
	> 	<mi>&epsilon;</mi>
	> 	<mo>]</mo>
	>	</math>

	>	在训练第一个弱分类器时：<math><mi>&epsilon;</mi><mo>=</mo><mi>分错的样本数</mi><mo>/</mo><mi>总样本数</mi></math> 
	
	>	在训练后续的弱分类器时：<math><mi>&epsilon;</mi></math> 为分错样本在D向量中的权值之和，因此也受到前几个弱分类器训练的影响

5. 直观的图像

	![Boosting训练过程](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101022146051659.png)
	
	> 绿色的线表示目前取得的模型（模型是由前m次得到的模型合并得到的），虚线表示当前这次模型。每次分类的时候，会更关注分错的数据，上图中，红色和蓝色的点就是数据，点越大表示权重越高，看看右下角的图片，当m=150的时候，获取的模型已经几乎能够将红色和蓝色的点区分开了。（来自[博客machine-learning-boosting-and-gradient-boosting](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/02/machine-learning-boosting-and-gradient-boosting.html)）

### 使用

1. 在一个难数据集上应用AdaBoost、以及使用流程
2. 训练错误率，测试错误率随弱分类器数目的变化
3. 过拟合现象

### 非均衡分类问题

1. 正确率、召回率、ROC曲线
2. AUC（Area Unser the Curve，曲线下面积）
3. 绘制AUC曲线的[代码](https://github.com/fangkun/cmt_ml_in_action/blob/master/ch_07_AdaBoost/adaboost.py)
4. 用代价矩阵来解非均衡分类问题：为TP,FN,FP,TN（True/False,Positive/Negative的缩写）赋予不同的权重，以体现模型设计者对分类器的要求
	
	~~~
	* AdaBoost：基于代价函数来调整错误权重向量D
	* 朴素贝叶斯：选择具有最小期望代价（而不是最大概率）的类别作为最后结果
	* SVM：在代价函数中，对不同类别选择不同的参数C
	~~~
	
5. 用欠抽样（undersampling，删除样本）和过抽样（oversampling，复制样本）来解非均衡分类问题
	
	> 欠抽样：删除非罕见类别的样本，使其与罕见类别的样本数量相当（但要防止有价值信息从非罕见类别中消失）
	
	~~~
	* 选择离决策边界远的样本进行删除
	* 使用非罕见类别的欠抽样、与罕见类别的过抽样相混合的方法
	~~~
	
	> 过抽样：增加罕见类别的样本，使其与非罕见类别样本数量相当
	
	~~~
	* 复制已有样例
	* 加入与已有样例相似的点
	* 加入已有数据的差值点（可能会引起过拟合）
	~~~
	
### 参考资料
	
1. [机器学习实践](http://www.ituring.com.cn/book/1021) 第7章 AdaBoost
2. [博客machine-learning-boosting-and-gradient-boosting](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/02/machine-learning-boosting-and-gradient-boosting.html)
	
