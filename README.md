# MachineLearningDeepLearning
吴恩达老师的机器学习课程个人笔记，deeplearning.ai（吴恩达老师的深度学习课程笔记及资源），吴恩达机器学习深度学习课后作业练习Python实现，李航《统计学习方法》的Python代码实现，附带一些个人笔记。GitHub上有很多吴恩达课程的代码资源，我也准备自己实现一下，后续会更新笔记，代码和百度云网盘链接。深度学习人工智能大数据吴恩达Andrew Ng      

Min's blog 欢迎访问我的博客主页！(Welcome to my blog website !)https://liweimin1996.github.io/

# 机器学习
# 深度学习
# 统计学习方法
# 自然语言处理

# 普通程序员如何正确学习人工智能方向的知识？
## 第一步：复习线性代数。
懒得看书就直接用了著名的——麻省理工公开课：线性代数，深入浅出效果拔群，以后会用到的SVD、希尔伯特空间等都有介绍；http://open.163.com/special/opencourse/daishu.html

## 入门机器学习算法。
还是因为比较懒，也就直接用了著名的——斯坦福大学公开课 ：机器学习课程，吴恩达教授的老版cs229的视频，讲的非常细（算法的目标->数学推演->伪代码）。这套教程唯一的缺点在于没有介绍最近大火的神经网络，但其实这也算是优点，让我明白了算法都有各自的应用领域，并不是所有问题都需要用神经网络来解决；http://open.163.com/special/opencourse/machinelearning.html

多说一点，这个课程里详细介绍的内容有：一般线性模型、高斯系列模型、SVM理论及实现、聚类算法以及EM算法的各种相关应用、PCA/ICA、学习理论、马尔可夫系列模型。课堂笔记在：CS 229: Machine Learning (Course handouts)，同样非常详细。
http://cs229.stanford.edu/materials.html

## 第三步：尝试用代码实现算法。
依然因为比较懒，继续直接使用了著名的——机器学习 | Coursera ，还是吴恩达教授的课程，只不过这个是极简版的cs229，几乎就是教怎么在matlab里快速实现一个模型（这套教程里有神经网络基本概念及实现）。这套课程的缺点是难度比较低，推导过程非常简略，但是这也是它的优点——让我专注于把理论转化成代码。
https://www.coursera.org/learn/machine-learning/home/welcome

## 第四步：自己实现功能完整的模型——进行中。
还是因为比较懒，搜到了cs231n的课程视频 CS231n Winter 2016 - YouTube ，李飞飞教授的课，主讲还有Andrej Karpathy和Justin Johnson，主要介绍卷积神经网络在图像识别/机器视觉领域的应用（前面神经网络的代码没写够？这门课包你嗨到爆~到处都是从零手写~）。这门课程的作业就更贴心了，直接用Jupyter Notebook布置的，可以本地运行并自己检查错误。主要使用Python以及Python系列的科学计算库（Scipy/Numpy/Matplotlib）。课堂笔记的翻译可以参考 智能单元 - 知乎专栏，主要由知友杜客翻译，写的非常好~
https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC
https://zhuanlan.zhihu.com/p/22339097

在多说一点，这门课对程序员来说比较走心，因为这个不像上一步中用matlab实现的作业那样偏向算法和模型，这门课用Python实现的模型同时注重软件工程，包括常见的封装layer的forward/backward、自定义组合layer、如何将layer组成网络、如何在网络中集成batch-normalization及dropout等功能、如何在复杂模型下做梯度检查等等；最后一个作业中还有手动实现RNN及其基友LSTM、编写有助于调试的CNN可视化功能、Google的DeepDream等等。（做完作业基本就可以看懂现在流行的各种图片风格变换程序了，如 cysmith/neural-style-tf）另外，这门课的作业实现非常推崇computational graph，不知道是不是我的幻觉……要注意的是讲师A.K的语速奇快无比，好在YouTube有自动生成解说词的功能，准确率还不错，可以当字幕看。
https://github.com/cysmith/neural-style-tf

因为最近手头有论文要撕，时间比较紧，第四步做完就先告一段落。后面打算做继续业界传奇Geoffrey Hinton教授的Neural Networks for Machine Learning | Coursera，再看看NLP的课程 Stanford University CS224d: Deep Learning for Natural Language Processing，先把基础补完，然后在东瞅瞅西逛逛看看有什么好玩的……
https://www.coursera.org/learn/neural-networks/home/welcome
http://cs224d.stanford.edu/

PS：一直没提诸如TensorFlow之类的神器，早就装了一个（可以直接在conda中为Tensorflow新建一个env，然后再装上Jupyter、sklearn等常用的库，把这些在学习和实践ML时所用到的库都放在一个环境下管理，会方便很多），然而一直没时间学习使用，还是打算先忍着把基础部分看完

关于用到的系统性知识，主要有：线性代数，非常重要，模型计算全靠它~一定要复习扎实，如果平常不用可能忘的比较多；高数+概率，这俩只要掌握基础就行了，比如积分和求导、各种分布、参数估计等等。（评论中有知友提到概率与数理统计的重要性，我举四肢赞成，因为cs229中几乎所有算法的推演都是从参数估计及其在概率模型中的意义起手的，参数的更新规则具有概率上的可解释性。对于算法的设计和改进工作，概统是核心课程，没有之一。
当拿到现成的算法时，仅需要概率基础知识就能看懂，然后需要比较多的线代知识才能让模型高效的跑起来。比如最近做卷积的作业， 我手写的比作业里给出的带各种trick的fast函数慢几个数量级，作业还安慰我不要在意效率，岂可修！）需要用到的编程知识也就是Matlab和Numpy了吧，Matlab是可以现学现卖的；至于Python，就看题主想用来做什么了，如果就是用来做机器学习，完全可以一天入门，如果想要做更多好玩的事，一天不行那就两天。

斯坦福大学2014（吴恩达）机器学习教程中文笔记

课程地址：https://www.coursera.org/course/ml

Machine Learning(机器学习)是研究计算机怎样模拟或实现人类的学习行为，以获取新的知识或技能，重新组织已有的知识结构使之不断改善自身的性能。它是人工智能的核心，是使计算机具有智能的根本途径，其应用遍及人工智能的各个领域，它主要使用归纳、综合而不是演译。在过去的十年中，机器学习帮助我们自动驾驶汽车，有效的语音识别，有效的网络搜索，并极大地提高了人类基因组的认识。机器学习是当今非常普遍，你可能会使用这一天几十倍而不自知。很多研究者也认为这是最好的人工智能的取得方式。在本课中，您将学习最有效的机器学习技术，并获得实践，让它们为自己的工作。更重要的是，你会不仅得到理论基础的学习，而且获得那些需要快速和强大的应用技术解决问题的实用技术。最后，你会学到一些硅谷利用机器学习和人工智能的最佳实践创新。

本课程提供了一个广泛的介绍机器学习、数据挖掘、统计模式识别的课程。主题包括：

（一）监督学习（参数/非参数算法，支持向量机，核函数，神经网络）。

（二）无监督学习（聚类，降维，推荐系统，深入学习推荐）。

（三）在机器学习的最佳实践（偏差/方差理论；在机器学习和人工智能创新过程）。本课程还将使用大量的案例研究，您还将学习如何运用学习算法构建智能机器人（感知，控制），文本的理解（Web搜索，反垃圾邮件），计算机视觉，医疗信息，音频，数据挖掘，和其他领域。

本课程需要10周共18节课，相对以前的机器学习视频，这个视频更加清晰，而且每课都有ppt课件，推荐学习。



正在学习林轩田的机器学习基石和吴恩达的机器学习，感觉讲的还不错，数学基础还是蛮重要的。
机器学习入门资源不完全汇总感谢贡献者： tang_Kaka_back@新浪微博欢迎补充指正，转载请保留原作者和原文链接。 本文是 机器学习日报的一个专题合集，欢迎订阅：请给hao@memect.com发邮件，标题＂订阅机器学习日报＂。机器学习入门资源不完全汇总基本概念机器学习 机器学习是近20多年兴起的一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。机器学习理论主要是设计和分析一些让计算机可以自动“学习”的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。因为学习算法中涉及了大量的统计学理论，机器学习与统计推断学联系尤为密切，也被称为统计学习理论。算法设计方面，机器学习理论关注可以实现的，行之有效的学习算法。下面从微观到宏观试着梳理一下机器学习的范畴：一个具体的算法，领域进一步细分，实战应用场景，与其他领域的关系。图1: 机器学习的例子：NLTK监督学习的工作流程图 (source: http://www.nltk.org/book/ch06.html)图2: 机器学习概要图 by Yaser Abu-Mostafa (Caltech) (source: Map of Machine Learning (Abu-Mostafa))图3: 机器学习实战：在python scikit learn 中选择机器学习算法 by Nishant Chandra (source: In pursuit of happiness!: Picking the right Machine Learning Algorithm)图4: 机器学习和其他学科的关系： 数据科学的地铁图 by Swami Chandrasekaran (source: Becoming a Data Scientist)机器学习入门资源不完全汇总入门攻略大致分三类： 起步体悟，实战笔记，行家导读机器学习入门者学习指南 @果壳网 (2013) 作者 白马 -- [起步体悟] 研究生型入门者的亲身经历有没有做机器学习的哥们？能否介绍一下是如何起步的 @ourcoders -- [起步体悟] 研究生型入门者的亲身经历，尤其要看reyoung的建议tornadomeet 机器学习 笔记 (2013) -- [实战笔记] 学霸的学习笔记，看看小伙伴是怎样一步一步地掌握“机器学习”Machine Learning Roadmap: Your Self-Study Guide to Machine Learning (2014) Jason Brownlee -- [行家导读] 虽然是英文版，但非常容易读懂。对Beginner,Novice,Intermediate,Advanced读者都有覆盖。A Tour of Machine Learning Algorithms （2013） 这篇关于机器学习算法分类的文章也非常好Best Machine Learning Resources for Getting Started（2013） 这片有中文翻译 机器学习的最佳入门学习资源 @伯乐在线 译者 programmer_lin门主的几个建议既要有数学基础，也要编程实践别怕英文版，你不懂的大多是专业名词，将来不论写文章还是读文档都是英文为主[我是小广告][我是小广告]订阅机器学习日报，跟踪业内热点资料。机器学习入门资源不完全汇总更多攻略机器学习该怎么入门 @知乎 (2014)What's the easiest way to learn machine learning @quora (2013)What is the best way to study machine learning @quora (2012)Is there any roadmap for learning Machine Learning (ML) and its related courses at CMU Is there any roadmap for learning Machine Learning (ML) and its related courses at CMU(2014)机器学习入门资源不完全汇总课程资源Tom Mitchell 和 Andrew Ng 的课都很适合入门机器学习入门资源不完全汇总入门课程机器学习入门资源不完全汇总2011 Tom Mitchell(CMU)机器学习英文原版视频与课件PDF 他的《机器学习》在很多课程上被选做教材，有中文版。Decision TreesProbability and EstimationNaive BayesLogistic RegressionLinear RegressionPractical Issues: Feature selection，Overfitting ...Graphical models: Bayes networks, EM，Mixture of Gaussians clustering ...Computational Learning Theory: PAC Learning, Mistake bounds ...Semi-Supervised LearningHidden Markov ModelsNeural NetworksLearning Representations: PCA, Deep belief networks, ICA, CCA ...Kernel Methods and SVMActive LearningReinforcement Learning 以上为课程标题节选机器学习入门资源不完全汇总2014 Andrew Ng (Stanford)机器学习英文原版视频 这就是针对自学而设计的，免费还有修课认证。“老师讲的是深入浅出，不用太担心数学方面的东西。而且作业也非常适合入门者，都是设计好的程序框架，有作业指南，根据作业指南填写该完成的部分就行。”（参见白马同学的入门攻略）"推荐报名，跟着上课，做课后习题和期末考试。(因为只看不干，啥都学不会)。" (参见reyoung的建议）Introduction (Week 1)Linear Regression with One Variable (Week 1)Linear Algebra Review (Week 1, Optional)Linear Regression with Multiple Variables (Week 2)Octave Tutorial (Week 2)Logistic Regression (Week 3)Regularization (Week 3)Neural Networks: Representation (Week 4)Neural Networks: Learning (Week 5)Advice for Applying Machine Learning (Week 6)Machine Learning System Design (Week 6)Support Vector Machines (Week 7)Clustering (Week 8)Dimensionality Reduction (Week 8)Anomaly Detection (Week 9)Recommender Systems (Week 9)Large Scale Machine Learning (Week 10)Application Example: Photo OCRConclusion机器学习入门资源不完全汇总进阶课程2013年Yaser Abu-Mostafa (Caltech) Learning from Data -- 内容更适合进阶 课程视频,课件PDF@CaltechThe Learning ProblemIs Learning Feasible?The Linear Model IError and NoiseTraining versus TestingTheory of GeneralizationThe VC DimensionBias-Variance TradeoffThe Linear Model IINeural NetworksOverfittingRegularizationValidationSupport Vector MachinesKernel MethodsRadial Basis FunctionsThree Learning PrinciplesEpilogue2014年 林軒田(国立台湾大学) 機器學習基石 (Machine Learning Foundations) -- 内容更适合进阶，華文的教學講解 课程主页When Can Machines Learn? [何時可以使用機器學習] The Learning Problem [機器學習問題] -- Learning to Answer Yes/No [二元分類] -- Types of Learning [各式機器學習問題] -- Feasibility of Learning [機器學習的可行性]Why Can Machines Learn? [為什麼機器可以學習] -- Training versus Testing [訓練與測試] -- Theory of Generalization [舉一反三的一般化理論] -- The VC Dimension [VC 維度] -- Noise and Error [雜訊一錯誤]How Can Machines Learn? [機器可以怎麼樣學習] -- Linear Regression [線性迴歸] -- Linear `Soft' Classification [軟性的線性分類] -- Linear Classification beyond Yes/No [二元分類以外的分類問題] -- Nonlinear Transformation [非線性轉換]How Can Machines Learn Better? [機器可以怎麼樣學得更好] -- Hazard of Overfitting [過度訓練的危險] -- Preventing Overfitting I: Regularization [避免過度訓練一：控制調適] -- Preventing Overfitting II: Validation [避免過度訓練二：自我檢測] -- Three Learning Principles [三個機器學習的重要原則]机器学习入门资源不完全汇总更多选择2008年Andrew Ng CS229 机器学习 -- 这组视频有些年头了，主讲人这两年也高大上了.当然基本方法没有太大变化，所以课件PDF可下载是优点。 中文字幕视频@网易公开课 | 英文版视频@youtube |课件PDF@Stanford第1集.机器学习的动机与应用 第2集.监督学习应用.梯度下降 第3集.欠拟合与过拟合的概念 第4集.牛顿方法 第5集.生成学习算法 第6集.朴素贝叶斯算法 第7集.最优间隔分类器问题 第8集.顺序最小优化算法 第9集.经验风险最小化 第10集.特征选择 第11集.贝叶斯统计正则化 第12集.K-means算法 第13集.高斯混合模型 第14集.主成分分析法 第15集.奇异值分解 第16集.马尔可夫决策过程 第17集.离散与维数灾难 第18集.线性二次型调节控制 第19集.微分动态规划 第20集.策略搜索2012年余凯(百度)张潼(Rutgers) 机器学习公开课 -- 内容更适合进阶 课程主页@百度文库 ｜ 课件PDF@龙星计划第1节Introduction to ML and review of linear algebra, probability, statistics (kai) 第2节linear model (tong) 第3节overfitting and regularization(tong) 第4节linear classification (kai) 第5节basis expansion and kernelmethods (kai) 第6节model selection and evaluation(kai) 第7节model combination (tong) 第8节boosting and bagging (tong) 第9节overview of learning theory(tong) 第10节optimization in machinelearning (tong) 第11节online learning (tong) 第12节sparsity models (tong) 第13节introduction to graphicalmodels (kai) 第14节structured learning (kai) 第15节feature learning and deeplearning (kai) 第16节transfer learning and semi supervised learning (kai) 第17节matrix factorization and recommendations (kai) 第18节learning on images (kai) 第19节learning on the web (tong)机器学习入门资源不完全汇总论坛网站机器学习入门资源不完全汇总中文我爱机器学习 我爱机器学习http://www.mitbbs.com/bbsdoc/DataSciences.html MITBBS－ 电脑网络 - 数据科学版机器学习小组 果壳 > 机器学习小组http://cos.name/cn/forum/22 统计之都 » 统计学世界 » 数据挖掘和机器学习北邮人论坛-北邮人的温馨家园 北邮人论坛 >> 学术科技 >> 机器学习与数据挖掘机器学习入门资源不完全汇总英文josephmisiti/awesome-machine-learning · GitHub 机器学习资源大全Machine Learning Video Library Caltech 机器学习视频教程库，每个课题一个视频Analytics, Data Mining, and Data Science 数据挖掘名站http://www.datasciencecentral.com/ 数据科学中心网站机器学习入门资源不完全汇总东拉西扯一些好东西，入门前未必看得懂，要等学有小成时再看才能体会。机器学习与数据挖掘的区别机器学习关注从训练数据中学到已知属性进行预测数据挖掘侧重从数据中发现未知属性Dan Levin, What is the difference between statistics, machine learning, AI and data mining?If there are up to 3 variables, it is statistics.If the problem is NP-complete, it is machine learning.If the problem is PSPACE-complete, it is AI.If you don't know what is PSPACE-complete, it is data mining.几篇高屋建瓴的机器学习领域概论, 参见原文The Discipline of Machine LearningTom Mitchell 当年为在CMU建立机器学习系给校长写的东西。A Few Useful Things to Know about Machine Learning Pedro Domingos教授的大道理，也许入门时很多概念还不明白，上完公开课后一定要再读一遍。几本好书李航博士的《统计学习方法》一书前段也推荐过，给个豆瓣的链接




