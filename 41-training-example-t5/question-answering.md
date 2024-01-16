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
## 3.1 绝对位置编码
### 3.1.1 三角函数式(Sinusoidal)位置编码
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;三角函数式(Sinusoidal)位置编码是在原Transformer模型中使用的一种显式编码。以一维三角函数编码为例：<br>

$$p_{k, 2i} = sin (\frac{k}{10000^{2 i / d}})$$

$$p_{k, 2i+1} = cos (\frac{k}{10000^{2 i / d}})$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一个长度为32 的输入序列（每个输入向量的特征维度是128）的Sinusoidal编码的可视化如下：<br>

![images](images/figure1.jpg)

### 3.1.2 可学习(Learnable)的位置编码
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;可学习(Learnable)位置编码是指将位置编码当作可训练参数，比如输入序列(经过嵌入层后)的大小为  $n \times d$  ，则随机初始化一个  $p \in \mathbb{R}^{n \times d}$  的矩阵作为位置编码，随训练过程更新。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;可学习位置编码的缺点是没有外推性，即如果预训练序列的最大长度为  n  ，则无法处理长度超过  n  的序列。此时可以将超过  n  部分的位置编码随机初始化并微调。<br>


## 3.2 相对位置编码
## 3.2.1 经典的相对位置编码
- [论文链接](https://aclanthology.org/N18-2074.pdf)


## 3.2.2 T5 中的相对位置编码
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

- [参考链接](https://kexue.fm/archives/8265)


# 4 参考链接
- [参考链接1](https://mp.weixin.qq.com/s/ENpXBYQ4hfdTLSXBIoF00Q)
- [参考链接2](https://www.cnblogs.com/shiyublog/p/11236212.html)
- [参考链接3](https://blog.nghuyong.top/2023/09/02/NLP/llm-position-embedding/)
