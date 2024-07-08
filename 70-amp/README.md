# 1. AMP 介绍
- [论文](https://github.com/Elvin-Ma/ai_papers/blob/main/mixed_precision/mixed-precision.md)

# 2. pytorch 实现
```python
import torch
from torch.cuda.amp import autocast as autocast

# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
```

# 3 fp16 训练的优势和不足
## 3.1 使用FP16训练神经网络，相对比使用FP32带来的优点
- 减少内存占用：FP16的位宽是FP32的一半，因此权重等参数所占用的内存也是原来的一半，节省下来的内存可以放更大的网络模型或者使用更多的数据进行训练。
- 加快通讯效率：针对分布式训练，特别是在大模型训练的过程中，通讯的开销制约了网络模型训练的整体性能，通讯的位宽少了意味着可以提升通讯性能，减少等待时间，加快数据的流通。
- 计算效率更高：在特殊的AI加速芯片如华为Ascend 910和310系列，或者NVIDIA VOTAL架构的Titan V and Tesla V100的GPU上，使用FP16的执行运算性能比FP32更加快。

## 3.2 FP16训练的问题
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;但是使用FP16同样会带来一些问题，其中最重要的是1）精度溢出和2）舍入误差。<br>
- 数据溢出：数据溢出比较好理解，FP16的有效数据表示范围为 6.10×10−5∼65504 ，FP32的有效数据表示范围为 1.4×10−45 1.7×1038 。可见FP16相比FP32的有效范围要窄很多，使用FP16替换FP32会出现上溢（Overflow）和下溢（Underflow）的情况。而在深度学习中，需要计算网络模型中权重的梯度（一阶导数），因此梯度会比权重值更加小，往往容易出现下溢情况。<br>
- 舍入误差：Rounding Error指示是当网络模型的反向梯度很小，一般FP32能够表示，但是转换到FP16会小于当前区间内的最小间隔，会导致数据溢出。如0.00006666666在FP32中能正常表示，转换到FP16后会表示成为0.000067，不满足FP16最小间隔的数会强制舍入。<br>

# 4. loss 进行 scale 缩放
## 4.1 损失缩放基本原理
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;损失放大是需要结合混合精度实现的，其主要的主要思路是：<br>
1. Scale up阶段，网络模型前向计算后在反响传播前，将得到的损失变化值DLoss增大2^K倍。
2. Scale down阶段，反向传播后，将权重梯度缩2^K倍，恢复FP32值进行存储。
  
## 4.2 动态损失缩放（Dynamic Loss Scaling）：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;上面提到的损失缩放都是使用一个默认值对损失值进行缩放，为了充分利用FP16的动态范围，可以更好地缓解舍入误差，**尽量使用比较大的放大倍数**。总结动态损失缩放算法，就是每当梯度溢出时候减少损失缩放规模，并且间歇性地尝试增加损失规模，从而实现在不引起溢出的情况下使用最高损失缩放因子，更好地恢复精度。<br>

## 4.3 动态损失缩放的算法
1. 动态损失缩放的算法会从比较高的缩放因子开始（如2^24），然后开始进行训练迭代中检查数是否会溢出（Infs/Nans）；
2. 如果没有梯度溢出，则不进行缩放，继续进行迭代；如果检测到梯度溢出，则缩放因子会减半，重新确认梯度更新情况，直到数不产生溢出的范围内；
3. 在训练的后期，loss已经趋近收敛稳定，梯度更新的幅度往往小了，这个时候可以允许更高的损失缩放因子来再次防止数据下溢。
4. 因此，动态损失缩放算法会尝试在每N（N=2000）次迭代将损失缩放增加F倍数，然后执行步骤2检查是否溢出。

