# 1 代码介绍

[代码仓库](https://github.com/huggingface/diffusers)


# 2 代码复现步骤

```shell
git clone https://github.com/huggingface/diffusers.git
```

# 3 stable diffusion 理论

![参考链接](http://www.zh0ngtian.tech/posts/c04f0a05.html)

# 5 评价指标
## 5.1 clip score
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Clip score是指将文本和图像对输入到OpenAI的CLIP（Contrastive Language-Image Pre-training）模型后分别转换为特征向量，然后计算它们之间的余弦相似度。当CLIP Score较高时，图像-文本对之间的相关性更高。CLIP Score评估自然语言(Promote 文本)和图像对之间的匹配度和相关性。值越大（接近1），评估越高。<br> ，

$$CLIP-SCORE(\mathbf{c}, \mathbf{v})=w * \max (\cos (\mathbf{c}, \mathbf{v}), 0)$$

![论文链接](https://aclanthology.org/2021.emnlp-main.595v2.pdf)

## 5.2 FID
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FID(Fréchet Inception Distance) score 是一种用于评估生成图像质量的度量标准，专门用于评估模型生成图片的性能，FID可以衡量生成图像的逼真度(image fidelity), 计算公式如下所示：<br>

$$FID(p, q)=||\mu_{p}-\mu_{q}||^{2} + Tr(C_{p} + C_{q} - 2 \sqrt{C_{p} C_{q}})$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其中, p 表示真实图像的分布, q 表示生成图像的分布, $\mu_{p}$  和  $\mu_{q}$  分别表示两个分布的特征向量的均值, $C_{p}$  和  $C_{q}$ 分别表示两个分布的特征向量的协方差矩阵。Tr 表示矩阵的迹运算, $|| \cdot\ ||_{2}$  表示欧几里得范数。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;具体来说, FID首先用 Inception network提取真实数据和生成数据的特征向量, 然后计算这两个特征向量集合的均值  $\mu_{1}$ , $\mu_{2}$  和协方差矩阵  $\Sigma_{1}$ ,  $\Sigma_{2}$ 。最后计算上述公式得到 FID值。<br>

FID 值越低代表两个分布越相似,生成的数据与真实数据分布越相似。<br>

