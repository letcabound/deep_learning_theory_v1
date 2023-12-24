# 1 参数初始化概念(Parameters initialization)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;参数初始化(parameters initialization)又称权重初始化(weight initialization), 具体指的是在网络模型训练之前，对各个节点的权重（weight）和偏置（bias）进行初始化赋值的过程。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;深度学习模型训练过程的本质是对 weight（即参数 W）进行更新，但是在最开始训练的时候需要每个参数有相应的初始值，这样神经网络就可以对权重参数w不停地迭代更新，以达到较好的性能。<br>

# 2 参数初始化的重要性
## 2.1 为什么参数初始化很重要
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;直观来看，在训练过程中，初始权重直接影响模型的输出，进一步影响损失及回传的梯度。因此，恰当的权重初始化是非常重要的。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;神经网络模型一般依靠随机梯度下降进行模型训练和参数更新，网络的最终性能与收敛得到的最优解直接相关，而收敛结果实际上又很大程度取决于网络参数的最开始的初始化。理想的网络参数初始化使模型训练事半功倍，相反，糟糕的初始化方案不仅会影响网络收敛，甚至会导致梯度弥散或爆炸。<br>

## 2.1 不合理初始化的问题
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果权值的初始值过大，则会导致梯度爆炸，使得网络不收敛；过小的权值初始值，则会导致梯度消失，会导致网络收敛缓慢或者收敛到局部极小值。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果权值的初始值过大，则loss function相对于权值参数的梯度值很大，每次利用梯度下降更新参数的时，参数更新的幅度也会很大，这就导致loss function的值在其最小值附近震荡。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;而过小的初始值则相反，loss关于权值参数的梯度很小，每次更新参数时，更新的幅度也很小，着就会导致loss的收敛很缓慢，或者在收敛到最小值前在某个局部的极小值收敛了。<br>

# 3 全0或常量初始化
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在神经网络中，把Parameters初始化为0是不可以的。这是因为如果把Parameters初始化，那么在前向传播过程中，每一层的神经元学到的东西都是一样的（激活值均为0），而在BP的时候，不同维度的参数会得到相同的更新，因为他们的gradient相同，这种行为称之为 **对称失效**。 <br>

# 4 随机初始化
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;随机初始化是很多人经常使用的方法，一般初始化的权重为高斯或均匀分布中随机抽取的值。然而这是有弊端的，一旦随机分布选择不当，就会导致网络优化陷入困境。<br>



# Xavier initialization

- [论文链接](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

# kaiming initialization

- [论文链接](https://arxiv.org/pdf/1502.01852)

# 参考文献
- [文献1](https://arxiv.org/pdf/1502.01852.pdf)
- [书籍](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html)
- [文献2](https://cloud.tencent.com/developer/article/1535198)
- [文献3](https://cloud.tencent.com/developer/column/5139)
