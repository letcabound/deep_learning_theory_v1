# 1 模型跑通
```shell
git clone https://github.com/huggingface/transformers.git

export PYTHONPATH=*/transformers/src:$PYTHONPATH

cd transformers

git checkout v4.31.0

cd */transformsers/examples/pytorch/question-answering

python run_seq2seq_qa.py \
  --model_name_or_path t5-small \
  --dataset_name squad_v2 \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_seq2seq_squad/
```

# 2 t5 介绍

- [t5 论文链接](https://arxiv.org/pdf/1910.10683.pdf)
- [t5 论文链接](https://links.jianshu.com/go?to=https%3A%2F%2Farxiv.org%2Fabs%2F1910.10683)

# 3 position embedding 总结
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;不同于RNN、CNN等模型，对于Transformer模型来说，位置编码的加入是必不可少的，因为纯粹的Attention模块是无法捕捉**输入顺序**的，即无法区分不同位置的Token。为此我们大体有两个选择：<br>
1. 想办法将位置信息融入到输入中，这构成了绝对位置编码的一般做法；<br>
2. 想办法微调一下Attention结构，使得它有能力分辨不同位置的Token，这构成了相对位置编码的一般做法。<br>

## 3.1 绝对位置编码
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一般来说，绝对位置编码会加到输入中：在输入的第 k 个向量  $x_{k}$ 中加入位置向量  $p_{k}$  变为  $x_{k} + p_{k}$ ，其中  $p_{k}$ 只依赖于位置编号k. <br>

- 绝对位置编码公式表达如下：<br>
![figure2](images/figure2.jpg)

### 3.1.1 三角函数式(Sinusoidal)位置编码
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;三角函数式(Sinusoidal)位置编码是在原 [Transformer论文](https://arxiv.org/abs/1706.03762) 中使用的一种显式编码。<br>

$$p_{k, 2i} = sin (\frac{k}{10000^{2 i / d}})$$

$$p_{k, 2i+1} = cos (\frac{k}{10000^{2 i / d}})$$

其中 $p_{k, 2 i}, p_{k, 2 i+1}$ 分别是位置 k 的编码向量的第 $2i, 2i+1$  个分量， d 是位置向量的维度。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;固定维度d为500，绘制不同N下的position embedding，具体如下：<br>

![figure1](images/figure1.jpg)

- 示例：<br>
![figure8](images/figure8.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;可以看到随着N的增大，周期会明显变长。文章中的N为10000，作者没有具体的解释，猜测可能是为了能让周期是一个很大的数，更好的区分开每个位置。<br>

### 3.1.2 可学习(Learnable)的位置编码
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;很显然，绝对位置编码的一个最朴素方案是不特意去设计什么，而是直接将位置编码当作可训练参数，比如最大长度为512，编码维度为768，那么就初始化一个512×768的矩阵作为位置向量，让它随着训练过程更新。现在的BERT、GPT等模型所用的就是这种位置编码.<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对于这种训练式的绝对位置编码，一般的认为它的缺点是没有外推性，即如果预训练最大长度为512的话，那么最多就只能处理长度为512的句子，再长就处理不了了。当然，也可以将超过512的位置向量随机初始化，然后继续微调。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;风极一时的 [bert](https://arxiv.org/abs/1810.04805) 中采用的就是这种编码方式，如下图所示：<br>

![figure7](images/figure7.jpg)

## 3.2 相对位置编码
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;相对位置并没有完整建模每个输入的位置信息，而是在算Attention的时候考虑当前位置与被Attention的位置的相对距离，由于自然语言一般更依赖于相对位置，所以相对位置编码通常也有着优秀的表现。对于相对位置编码来说，它的灵活性更大，更加体现出了研究人员的“天马行空”。<br>

### 3.2.1 经典的相对位置编码
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;相对位置编码起源于Google的论文[《Self-Attention with Relative Position Representations》](https://arxiv.org/abs/1803.02155)，一般认为，相对位置编码是由绝对位置编码启发而来，我们再回忆下一般的带绝对位置编码的Attention：<br>

![figure2](images/figure2.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其中 softmax 对 j 那一维归一化，这里的向量都是指行向量。我们初步展开 $k_{j}^{\top}$ . <br>

![figure9](images/figure9.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了引入相对位置信息，Google把第一项位置去掉，第二项 $p_{j} W_{K}$ 改为二元位置向量 $R_{i, j}^{K}$ . 变成： <br>

$$a_{i, j}=softmax(x_{i} W_{Q}(x_{j} W_{K}+R_{i, j}^{K})^{\top})$$

以及 $o_{i}=\sum_{j} a_{i, j} v_{j} = \sum_{j} a_{i, j}(x_{j} W_{V}+p_{j} W_{V})$ 中的 $p_{j} W_{V}$ 换成 $R_{i, j}^{V}$ : <br>

$$o_{i}=\sum_{j} a_{i, j}(x_{j} W_{V}+R_{i, j}^{V})$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;所谓相对位置，是将本来依赖于二元坐标  (i, j)  的向量  $R_{i, j}^{K}, R_{i, j}^{V}$ ，改为只依赖于相对距离  i-j，并且通常来说会进行截断，以适应不同任意的距离: <br>

$$R_{i, j}^{K}=p_{K}[clip(i-j, p_{\min }, p_{\max })]$$

$$R_{i, j}^{v}=p_{v}[clip(i-j, p_{\min }, p_{\max })]$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;这样一来，只需要有限个位置编码，就可以表达出任意长度的相对位置（因为进行了截断），不管 $p_{K} ,p_{V}$ 是选择可训练式的还是三角函数式的，都可以达到处理任意长度文本的需求。<br>

### 3.2.2 T5 中的相对位置编码

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;T5模型出自文章[《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》](https://arxiv.org/abs/1910.10683)，里边用到了一种更简单的相对位置编码。<br>

将


- [来源](https://arxiv.org/abs/1910.10683)

- 形成相对位置坐标
```python
a = torch.arange(15)[:, None]
b = torch.arange(15)[None, :]
c = a -b
>>> c.shape
torch.Size([15, 15])
>>> c
tensor([[  0,  -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9, -10, -11, -12, -13,
         -14],
        [  1,   0,  -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9, -10, -11, -12,
         -13],
        [  2,   1,   0,  -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9, -10, -11,
         -12],
        [  3,   2,   1,   0,  -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9, -10,
         -11],
        [  4,   3,   2,   1,   0,  -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,  -9,
         -10],
        [  5,   4,   3,   2,   1,   0,  -1,  -2,  -3,  -4,  -5,  -6,  -7,  -8,
          -9],
        [  6,   5,   4,   3,   2,   1,   0,  -1,  -2,  -3,  -4,  -5,  -6,  -7,
          -8],
        [  7,   6,   5,   4,   3,   2,   1,   0,  -1,  -2,  -3,  -4,  -5,  -6,
          -7],
        [  8,   7,   6,   5,   4,   3,   2,   1,   0,  -1,  -2,  -3,  -4,  -5,
          -6],
        [  9,   8,   7,   6,   5,   4,   3,   2,   1,   0,  -1,  -2,  -3,  -4,
          -5],
        [ 10,   9,   8,   7,   6,   5,   4,   3,   2,   1,   0,  -1,  -2,  -3,
          -4],
        [ 11,  10,   9,   8,   7,   6,   5,   4,   3,   2,   1,   0,  -1,  -2,
          -3],
        [ 12,  11,  10,   9,   8,   7,   6,   5,   4,   3,   2,   1,   0,  -1,
          -2],
        [ 13,  12,  11,  10,   9,   8,   7,   6,   5,   4,   3,   2,   1,   0,
          -1],
        [ 14,  13,  12,  11,  10,   9,   8,   7,   6,   5,   4,   3,   2,   1,
           0]])
```

## 3.3 旋转位置编码
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;旋转位置编码（Rotary Position Embedding，RoPE）是论文 Roformer: Enhanced Transformer With Rotray Position Embedding 提出的一种能够将相对位置信息依赖集成到 self-attention 中并提升 transformer 架构性能的位置编码方式。而目前很火的 LLaMA、GLM 模型也是采用该位置编码方式。和相对位置编码相比，RoPE 具有更好的外推性，目前是大模型相对位置编码中应用最广的方式之一。<br>

**思考：什么是大模型外推性？** <br>

外推性是指大模型在训练时和预测时的输入长度不一致，导致模型的泛化能力下降的问题。例如，如果一个模型在训练时只使用了 512 个 token 的文本，那么在预测时如果输入超过 512 个 token，模型可能无法正确处理。这就限制了大模型在处理长文本或多轮对话等任务时的效果。<br>

- llama 中的RoPE 代码实现

```python
# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.wq = Linear(...)
        self.wk = Linear(...)
        self.wv = Linear(...)

        self.freqs_cis = precompute_freqs_cis(dim, max_seq_len * 2)

    def forward(self, x: torch.Tensor):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, dim)
        xk = xk.view(batch_size, seq_len, dim)
        xv = xv.view(batch_size, seq_len, dim)

        # attention 操作之前，应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # scores.shape = (bs, seqlen, seqlen)
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(dim)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, xv)  # (batch_size, seq_len, dim)
  # ......
```

- [参考链接](https://hub.baai.ac.cn/view/29979)
- [参考链接](https://kexue.fm/archives/8265)


# 4 参考链接
- [参考链接1](https://mp.weixin.qq.com/s/ENpXBYQ4hfdTLSXBIoF00Q)
- [参考链接2](https://www.cnblogs.com/shiyublog/p/11236212.html)
- [参考链接3](https://blog.nghuyong.top/2023/09/02/NLP/llm-position-embedding/)
- [参考链接4](https://juejin.cn/post/7126132489428402184)
- [参考链接5](https://https://kexue.fm/archives/8130)
