# 0 如何在多个 GPU 上训练非常大的模型？
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;近年来，我们看到许多自然语言处理基准任务利用更大的预训练语言模型取得更好的结果。如何训练大型深度神经网络具有挑战性，因为它需要大量的 GPU 内存和漫长的训练时间。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;然而，单个 GPU 工作器的**内存有限**，许多大型模型的尺寸已经超出单个 GPU 的范围。有几种并行范式可实现跨多个 GPU 进行模型训练，以及各种模型架构和节省内存设计，以帮助实现训练非常大型的神经网络。<br>

# 1 训练并行性
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;训练非常大型神经网络模型的主要瓶颈是对大量 GPU 内存的巨大需求，远远超出单个 GPU 机器的容量。除了模型权重（例如数十亿个浮点数）之外，通常存储中间计算输出（如梯度和优化器状态，如 Adam 中的动量和变化）的成本更高。此外，训练大型模型通常需要大量的训练语料库，因此单个进程可能需要很长时间。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;因此，并行性是必要的。并行性可以在不同的维度上发生，包括数据、模型架构和张量操作。<br>



# 参考文档
- [How to train really large models on many gpus](https://lilianweng.github.io/posts/2021-09-25-train-large/)
- [How to train really large models on many gpus](https://openai.com/index/techniques-for-training-large-neural-networks/)
