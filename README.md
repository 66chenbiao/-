网络资源收集
====



## 深度学习

[2020-11-8]
[高等深度学习][1]

[1]:https://github.com/PKUAI26/Deep-Learning-Advanced-Topics

## scikit-learn源码分析
整个SKlearn项目，核心的部分在sklearn这个包里面，算法的使用案例在example包里面
![scikit-learn 项目结构](https://img2020.cnblogs.com/blog/1342077/202003/1342077-20200306162555246-556833908.png)
## sklearn包的简介
* _check_build
简单的检查是否正确编译的脚本
* _build_utils
sklearn官方团队构建这个项目时使用的一些支持性工具
* _loss
GLM算法中会用到的分布函数
* cluster
聚类算法的实现包。含有Birch/DBSCAN/Hierarchical/KMeans/Spectral等算法，其中kmeans算法提供elkan/fast/lloyd几种实现方式(使用cpyton实现的)。
* compose
合成模型时使用的元学习器。
* covariance
计算特征间协方差
* cross_decomposition
包含CCA（典型相关分析）和PLS（偏最小二乘）两种算法，这些算法主要用于探索两个多元数据集之间的线型关系。
* datasets
sklearn自带的数据采集器，主要功能时响应用户的调用，从网络下载玩具数据集，方便用户简单地跑跑算法。
* decompostion
矩阵分解算法包。包括PLA（主成分分析）、NMF（非负矩阵分解）、ICA（独立成分分析），其中PLA有稀疏版本的实现。
* ensemble
集成算法包。主要的脚本有：_bagging.py(bagging算法),_forest.py(随机森林),_gb(GBDT),_iforest.py(孤独森林)，_stacking.py（stacking方法）,_voting.py(投票方法),_weight_boosting.py(主要就是AdaBoost算法)。注意，GBDT算法的实现过程有调用一个cpython版本的脚本_gradient_boosting.pyx，主要目的就是加快计算速度。
* experimental
实验模块
* externals
一些外部依赖脚本。
* feature_extraction
特征提取。目前支持同Text文档和图片中提取特征。
* feature_selection
特征选择算法，主要是单变量的过滤算法（例如去除方差很小的特征）和递归特征删除算法。这里的接口也可以用来做降维。
* guassian_process
高斯过程。分为分类和回归两个实现。
* impute
缺失值处理，例如使用KNN算法去填充缺失值。
* inspection
检查模型的工具
* linear_model
线性模型包，包含线性回归、逻辑回归、LASSO、Ridge、感知机等等，内容非常多，是sklearn中的重点包
* manifold
实现数据嵌入技术的工具包，包括LLE、Iosmap、TSNE等等
* metrics
集合了所有的度量工具，包括常见的accuracy_socre、auc、f1_score、hinge_loss、roc_auc_score、roc_curve等等
* mixture
高斯混合模型、贝叶斯混合模型
* model_selection
sklearn的重点训练工具包，包括常见的GridSearchCV、TimeSeriesSplit、KFold、cross_validate等
* neighbors
K近邻算法包，包括球树、KD树、KNN分类、KNN回归等算法的实现
* neutral_network
包含比较基础的神经网络模型，例如伯努利受限玻尔兹曼机、多层感知机分类、多层感知机回归
* preprocessing
sklearn的重点数据预处理工具包，包括常见的LabelEncoder、MinMaxScaler、Normalizer、OneHotEncoder等
* semi_supervises
半监督学习算法包，LabelPropagation、LabelSpreading
* svm
支持向量机算法包，包括线性支持向量分类/回归、SVC/SVR、OneClassSVM等，但是sklearn自己并没有独立去实现这一类算法，而是复用了很多libSVM的代码
* tests
一些单元测试代码
* tree
树模型，包括决策树、极端树。注意，梯度提升树算法放在ensemble包里。
* utils
加速计算、cython版本的BLAS算法、优化等工具包

sklearn项目可以看成一棵大树，各种estimator是果实，而支撑这些估计器的主干，是为数不多的几个基类。常见的几个类有BaseEstimator、BaseSGD、ClassifierMixin、RegressorMixin，等等
官方文档的API参考页面列出了主要的API接口，我们看下Base类
![](https://img2020.cnblogs.com/blog/1342077/202003/1342077-20200320212850771-1859142633.png)
## BaseEstimator
最底层的就是BaseEstimator类。主要暴露两个方法：set_params，get_params

### get_params函数细节
这个方法旨在获取对象的参数，返回对象默认是{参数:参数值}的键值对。如果将get_params的参数deep设置为True，还会返回（如果有的话）子对象（它们是估计器）。下面我们来仔细看一下这个方法的实现细节
![](https://img2020.cnblogs.com/blog/1342077/202003/1342077-20200320165119520-70294856.png)


1.[简单实现的scikit-learn neighbors][2]

2.[sklearn 源码分析系列：neighbors][3]


[2]:https://github.com/demonSong/DML
[3]:https://blog.csdn.net/u014688145/article/details/61916582
