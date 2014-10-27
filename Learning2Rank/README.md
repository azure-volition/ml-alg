## Learning To Rank

### 资料列表

* 应用背景：龙星计划2012，机器学习 Lecture 19 
	
* [WIKI](http://en.wikipedia.org/wiki/Learning_to_rank)：应用，特征处理，方法，论文，实现代码

* [Yahoo Learning to Rank Challenge Overview [2011]](../ml-pdf/yahoo_chapelle11a.pdf)：数据集、算法和论文列表

* [Survey [2009]]()：Learning to Rank Survey: T. Y. Liu. Learning to rank for information retrieval. Foundations and Trends in Informa- tion Retrieval, 3(3):225–331, 2009


### GBRank: boosted tree based on preference learning
	
* Zhaohui Zheng, Hongyuan Zha, Keke Chen, Gordon Sun. [A Regression Frame work for Learning Ranking Functions Using Relative Relevance Judgments](../ml-pdf/fp086_zheng_2007gbrank.pdf) In SIGIR 2007, pages 287-294, 2007.

* 其他相关：
	
	> Z. Zheng, H. Zha, T. Zhang, O. Chapelle, K. Chen, and G. Sun. [A general boosting method and its application to learning ranking functions for web search](../ml-pdf/nips07_gbrank.pdf). In Advances in Neural Information Processing Systems 20, pages 1697–1704. MIT Press, 2008. 

### RankSVM: SVM based on preference learning

* 最初论文：R. Herbrich, T. Graepel, and K. Obermayer. [Large margin rank boundaries for ordinalVregression](../ml-pdf/RankSVM1999_herobergrae99.pdf). In Smola, Bartlett, Schoelkopf, and Schuurmans, editors, Advances in Large Margin Classiers. MIT Press, Cambridge, MA, 2000.

* 更新一些的一篇，介绍了一个具体的应用（用click through logs作为样本）：

	> T. Joachims. [Optimizing search engines using clickthrough data](../ml-pdf/RankSVM2002_joachims_02c.pdf). In Proceedings of the ACM Conference on Knowledge Discovery and Data Mining (KDD). ACM, 2002.

* 可用（public available）的算法及代码：

	> RankSVM We considered the linear version of this pairwise ranking algorithm first in- troduced in (Herbrich et al., 2000). There are several efficient implementation of this algorithm that are publicly available ([Joachims, 2006](../ml-pdf/RankSVM2006_joachims_06a.pdf); [Chapelle and Keerthi, 2010](../ml-pdf/RankSVM2010_ordinal.pdf)). We used the former code available at [http://www.cs.cornell.edu/People/tj/svm_light/svm_rank.html](http://www.cs.cornell.edu/People/tj/svm_light/svm_rank.html). 

	> 摘自[Yahoo Learning to Rank Challenge Overview [2011]](../ml-pdf/yahoo_chapelle11a.pdf)



