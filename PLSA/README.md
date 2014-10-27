<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


	
## PLSA（Probabilistic latent semantic analysis）

### 相关知识

[Topic Model](http://en.wikipedia.org/wiki/Topic_model)

[Generative model](http://en.wikipedia.org/wiki/Generative_model)

[Bayesian Network](http://en.wikipedia.org/wiki/Bayesian_network)

### 快速了解

[博客](http://blog.tomtung.com/2011/10/plsa/)

[博客](http://blog.csdn.net/yangliuy/article/details/8330640)

### 课程

[龙星2010机器学习 Lecture14 Latent Aspect Models]() (TODO)

### 使用

* 输入：文档库（M篇文档），
<!--没有有文档主题标注???-->

* 输出：K个主题，每篇文档可能属于哪几个主题，概率是多少

* 预测结果的影响因素：
	* 主题划分粗细程度
	* 主题内文章的数量

### 模型原理

* 样本表示：
	
	* 每个单词在每篇文档中出现的次数 n(w,d)（其中w&isin;W，d&isin;D，W是单词集合，D是文档集合）
	
		> PLSA只考虑文档中单词出现次数（忽略词出现的先后顺序），单词之间是独立的（bag-of-words假设）

	* 样本集中总共有M篇文档，N个单词，第j篇文档的单词数为 N<sub>j</sub>
	* 假设每对(d,w)都对应一个隐藏的主题z（其中z&isin;Z）
	* 用[贝叶斯网络](http://en.wikipedia.org/wiki/Bayesian_network)表示，[图示](http://en.wikipedia.org/wiki/Plate_notation)如下：
	
		> ![](http://blog.tomtung.com/images/2011-10-19-plsa_graph.png)
	
* 生成(d,w)对，每个(d<sub>i</sub>,w<sub>j</sub>)对的生成方法如下：
	* 计算P(d<sub>i</sub>)，即单词w出现在文档d<sub>i</sub>中的概率，以此概率选中文档d<sub>i</sub>
	* 计算P(z<sub>k</sub>|d<sub>i</sub>)，即文档d<sub>i</sub>中出现主题z<sub>k</sub>的概率（每篇文章中各主题的占比），以此概率选中主题z<sub>k</sub>
	* 计算P(w<sub>j</sub>|z<sub>k</sub>)，即主题z<sub>k</sub>中出现单词w<sub>j</sub>的概率（每个主题中各单词出现的概率），以此概率产生一个单词w<sub>j</sub>
	
	<!-- 选择规则是什么? -->

* 计算(d,w)对的联合分布，对于每个(d<sub>i</sub>,w<sub>j</sub>)对，[联合概率](http://baike.baidu.com/view/2485096.htm?fr=aladdin)为：

	*  
<math display="block">
      <mi>P</mi><mo>(</mo><mi>d<sub>i</sub></mi><mo>,</mo><mi>w<sub>j</sub></mi><mo>)</mo>
      <mo>=</mo>
      <mi>P</mi><mo>(</mo><mi>d<sub>i</sub></mi><mo>)</mo>
      <mi>P</mi><mo>(</mo><mi>w<sub>j</sub></mi><mo>|</mo><mi>d<sub>i</sub></mi><mo>)</mo>
</math>
	
	
	* 
<math display="block">
	<mi>P</mi><mo>(</mo><mi>w<sub>j</sub></mi><mo>|</mo><mi>d<sub>i</sub></mi><mo>)</mo>
	<mo>=</mo>
	<mo>&sum;</mo>
	<mi>P</mi><mo>(</mo><mi>w<sub>j</sub></mi><mo>|</mo><mi>z<sub>k</sub></mi><mo>)</mo>
	<mi>P</mi><mo>(</mo><mi>z<sub>k</sub></mi><mo>|</mo><mi>d<sub>i</sub></mi><mo>)</mo>
</math>&nbsp;&nbsp;for all k&isin;K
	<p>这样的概率要计算|Z|.|D|+|W|.|Z|个</p>
	
	* 将计算结果用&theta;表示，就是我们希望估计的模型参数： 
	
		> &theta;=(P(z|d),P(w|z)) &nbsp;&nbsp;for all (z<sub>k</sub>, d<sub>i</sub>)&isin;(Z,D), (w<sub>j</sub>,z<sub>k</sub>)&isin;(W,Z)
	
		> 根据[最大log似然估计法](http://blog.csdn.net/yangliuy/article/details/8296481)，希望找到参数&theta;，使得&nbsp;&nbsp;&sum;log(P(d,w)<sup>n(d,w)</sup>)&nbsp;&nbsp;&nbsp;&nbsp;(for all d&isin;D, w&isin;W)&nbsp;&nbsp;最大

* 求解 &theta;

	* 因为 &theta; 中的隐藏变量 z&isin;Z 无法被观察到，因此需要借助[EM算法](http://blog.tomtung.com/2011/10/em-algorithm/)，求解过程及代码实例可参考 [http://blog.tomtung.com/2011/10/plsa/](http://blog.tomtung.com/2011/10/plsa/)



	








	
	
	
	
	
	