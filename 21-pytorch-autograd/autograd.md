# autograd


# 1 pytorch autograd(自动微分机制)
- [pytorch link](https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本文将概述自动微分（autograd）的工作原理和记录操作的方式。虽然不一定需要完全理解其中的所有内容，但我们建议您熟悉它，因为这将有助于您编写更高效、更清晰的程序，并可帮助您进行调试。<br>

## 1.1 自动微分如何编码历史记录
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;自动微分是一种反向自动微分系统。在概念上，自动微分在执行操作(operations)时记录了创建数据的所有操作(operations)，从而生成了一个**有向无环图**，其叶节点是输入张量，而根节点是输出张量。通过从根节点到叶节点追溯这个图，您可以使用链式法则自动计算梯度。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在内部，自动微分将这个图表示为一组函数对象(实际上是表达式)，可以通过 apply() 方法应用这些函数对象来计算图的求值结果。在计算前向传播时，自动微分同时执行请求的计算，并构建表示计算梯度的函数的图(每个 torch.Tensor 的 .grad_fn 属性是进入这个图的入口)。完成前向传播后，我们在反向传播中评估这个图，以计算梯度。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;需要注意的重要一点是，图在每次迭代时都是**从头开始重新创建的(动态性的一个重要体现)**，这正是允许使用任意的 Python 控制流语句的原因，这些语句可以在每次迭代时改变图的整体形状和大小。在启动训练之前，您不必编码所有可能的路径 - 您运行的就是您要求微分的内容。<br>

## 1.2 Saved Tensors
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在前向传播过程中，有些操作需要保存中间结果以便执行反向传播。例如，函数 x ↦ x² 保存输入 x 来计算梯度。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当定义一个自定义的 Python [Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function) 时，您可以使用 save_for_backward() 在前向传播期间保存张量，并使用 saved_tensors 在反向传播期间检索它们。有关更多信息，请参阅 [扩展 PyTorch](https://pytorch.org/docs/stable/notes/extending.html)。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对于 PyTorch 定义的操作（例如 torch.pow()），张量会根据需要自动保存。您可以探索（出于教育或调试目的）哪些张量由特定 grad_fn 保存，方法是查找以 _saved 为前缀的属性。<br>

```python
x = torch.randn(5, requires_grad=True)
y = x.pow(2)
print(x.equal(y.grad_fn._saved_self))  # True
print(x is y.grad_fn._saved_self)  # True
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在上述代码中，y.grad_fn._saved_self 引用的是与 x 相同的 Tensor 对象。但并非总是如此。例如：<br>

```python
x = torch.randn(5, requires_grad=True)
y = x.exp()
print(y.equal(y.grad_fn._saved_result))  # True
print(y is y.grad_fn._saved_result)  # False
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在内部，为了防止引用循环，PyTorch在保存时对张量进行了打包，并在读取时将其解包为不同的张量。在这里，从访问 y.grad_fn._saved_result 得到的张量与 y 是不同的张量对象（但它们仍然共享相同的存储）。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;张量是否会被打包为不同的张量对象取决于它是否是其自身 grad_fn 的输出，这是一个可能变化的实现细节，用户不应依赖此特性。<br>

您可以使用 [张量的钩子(Hooks)](https://pytorch.org/docs/stable/notes/autograd.html#saved-tensors-hooks-doc) 来控制 PyTorch 如何进行保存张量的打包/解包操作。

## 1.3 对于不可微分的函数的梯度计算
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用自动微分进行梯度计算仅在每个使用的基本函数都是可微分的情况下有效。不幸的是，我们在实践中使用的许多函数并不具备这个性质（例如 relu 或 sqrt 在 0 处）。为了尽量减少不可微分函数的影响，我们按照以下规则定义基本操作的梯度计算顺序：<br>

1. 如果该函数可微分，因此在当前点存在梯度，请使用它。<br>
2. 如果该函数是凸函数（至少在局部上），请使用模最小的子梯度（它是最陡下降方向）。<br>
3. 如果该函数是凹函数（至少在局部上），请使用模最小的超梯度（考虑 -f(x) 并应用上述点）。<br>
4. 如果该函数被定义，通过连续性在当前点定义梯度（请注意，这里可能存在无穷大，例如 sqrt(0)）。如果存在多个值，则任意选择一个。<br>
5. 如果该函数未定义（sqrt(-1)、log(-1) 或当输入为 NaN 时的大多数函数），那么作为梯度使用的值是任意的（我们也可能引发错误，但不能保证）。大多数函数将使用 NaN 作为梯度，但出于性能原因，某些函数将使用其他值（例如，log(-1)）。<br>
6. 如果该函数不是确定性映射（即它不是数学函数），则将其标记为不可微分。如果在需要梯度的张量上在 no_grad 环境之外使用它，这将导致在反向传播过程中引发错误。<br>

## 1.4 局部禁用梯度计算
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;有几种机制可以在 Python 中局部禁用梯度计算：

- 要在代码的整个块中禁用梯度，可以使用像 no-grad 模式和推断模式这样的上下文管理器。
- 为了更细粒度地排除子图不进行梯度计算，可以设置张量的 requires_grad 字段。
- 除了讨论上述机制之外，下面我们还会介绍评估模式（nn.Module.eval()），这是一种**不用于**禁用梯度计算的方法，但由于其名称，经常与前面的三种方法混淆。

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
with torch.no_grad():
    y = x * 2  # 这里的计算不会被跟踪梯度

with torch.set_grad_enabled(False):
    y = x * 2  # 这里的计算不会被跟踪梯度

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
x.requires_grad = False  # 禁用 x 的梯度计算
z = x * y  # 只有 y 的梯度会被计算
```

## 
