# weight initialization 概述
在训练过程中，初始权重直接影响模型的输出，进一步影响损失及回传的梯度。因此，恰当的权重初始化是非常重要的。<br>

# 1 常量初始化
权重全部初始化为0，梯度回传后全为0，多于一层的NN就无法学习。 梯度全部初始化为1，梯度回传值全部相同，一层内不同神经元的值全部一样，无意义。<br>

# Xavier initialization

- [论文链接](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

# kaiming initialization

- [论文链接](https://arxiv.org/pdf/1502.01852)

# 参考文献
- [文献1](https://arxiv.org/pdf/1502.01852.pdf)
- [书籍](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html)
- [文献2](https://cloud.tencent.com/developer/article/1535198)
- [文献3](https://cloud.tencent.com/developer/column/5139)
