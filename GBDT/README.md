## GBDT(Gradient Boost Decision Tree)

### 资料

> [博客决策树模型组合之随机森林与GBDT（快速理解）](http://www.cnblogs.com/LeftNotEasy/archive/2011/03/07/random-forest-and-gbdt.html)

> [WIKI](http://en.wikipedia.org/wiki/Gradient_boosting)

### 相关基础

> [决策树](http://www.autonlab.org/tutorials/dtree.html)

> [信息增益](http://www.autonlab.org/tutorials/infogain.html)


### 数学原理

> [机器学习中的数学(3)-模型组合(Model Combining)之Boosting与Gradient Boosting](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/02/machine-learning-boosting-and-gradient-boosting.html)

> [论文Greedy function Approximation – A Gradient Boosting Machine (Freidman,1999)](../ml-pdf/trebst.pdf)

参考上述两篇文章，相关注解如下：

模型公式

> h : 各个子模型的hypothesis function，模型训练参数是 &alpha;_i

> F : 是M个子模型hyhothesis function的加权求和，权值参数是 &beta;_i

> P : 模型训练目标，也就上h,F中的训练参数 &alpha;, &beta; 

> L : 是loss function。对与回归，可以是(y-F)^2或|y-F|；对与分类，可以是log(1+e^(-2yF))

> Ex,y: 用来计算L的期望值（x为样本，y为标注的分类/回归结果）

> &Theta;(P)=Ex,yL(y, F(x;P)): 样本集<x,y>上，模型的损失期望函数，也就是P的likelyhood函数

训练目标

> P*: 经过训练，得到P的最优值P * 可以让损失期望得到最低值 &Theta;(P *)，将新样本x代入，就可以得到预测结果 F*(x,P*)

模型训练（梯度下降），先假定F只包含1个hypthesis function

> P*=sum(p_i): 因为F和L都是可累加的，因此可以分解为一系列增量；在每一轮训练中，逐步施加增量进行调整，调整方法为Gradient Descent

> g_x：在第x轮迭代中，找到梯度下降的方向（对&Theta;在各个p_j上求偏导，得到方向向量），其中P{p_1,p_2,...} 是前x-1轮的累加结果

> &rho;_x： 下降步长

> P_x = &rho;_x * g_x：在 &rho;_x 方向上移动步长 g_x，以便能够最小化 &Theta;(P_m)

模型训练（函数空间推广），F由M个hypothesis function加权合成

> P不仅可以在被分解成梯度下降中各轮迭代的增量，同样也可以推广在函数空间中，表达为函数空间中各函数参数的累加值，在整个函数空间中做梯度下降

> F_m(x)=sum(f(x)) : 由函数空间中m个hypothesis function加权求和组成

> P_m(x)=sum(p_i)  : 由函数空间中m个hypothesis function的参数(&alpha;)，以及函数权值（&beta;）求和而成（ 理解为(&alpha;_1, &alpha;_2, ... , &alpha;_N) * &beta;_m ? ）

> 这样可以类似地采用梯度下降的方法

通用模型：综上所述

> hypthesis function：

> ![](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101022147109807.png)

> 训练目标：

> ![](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101022147253239.png)

> 在前m-1个子模型的基础上，确定第m个子模型梯度下降的方向( 参数 &alpha; 及权值 &beta;)

> ![](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101022147263140.png)

> ![](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101022147267251.png)

> ![](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101022147493412.png)

> 使用最小二乘法为上述 &alpha;, &beta; 求解

> ![](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101022147505821.png)

> ![](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101022147549852.png)

> 乘以下降步长，实施梯度下降，合并到模型中

> ![](http://images.cnblogs.com/cnblogs_com/LeftNotEasy/201101/201101022147575868.png)


### 算法

> Gradient Boosting 与传统Boosting（如：AdaBoosting）的区别是，每一轮迭代，是为了减少上一次的残差（residual）。

> 为了消除残差（residual），在残差减少的梯度(Gradient)方向上建立一个新的子模型，使之前的模型向残差梯度的方向减少。

> TODO: Multi-Class Logistic

> TODO: 读论文

### 开源实现

> [http://elf-project.sourceforge.net/](http://elf-project.sourceforge.net/)

