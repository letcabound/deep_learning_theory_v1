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

## 1.5 设置 requires_grad
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;requires_grad 是一个标志，除非包装在 nn.Parameter 中，默认为 false，它允许对子图进行细粒度的梯度计算排除。它在前向传播和反向传播中都起作用：<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在前向传播过程中，只有至少一个输入张量需要 grad 的操作才会在反向图中记录。在反向传播过程（.backward()）中，只有 requires_grad=True 的叶张量才会将梯度累积到其 .grad 字段中。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;重要的是要注意，尽管每个张量都有这个标志，但仅对叶张量设置它才有意义（没有 grad_fn 的张量，例如 nn.Module 的参数）。非叶张量（具有 grad_fn 的张量）是具有与之相关联的反向图的张量。因此，它们的梯度将作为计算需要，作为计算需要 grad 的叶张量的中间结果。根据这个定义，所有非叶张量将自动具有 require_grad=True。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;设置 requires_grad 应该是控制模型中哪些部分参与梯度计算的主要方法，例如，在模型微调期间需要冻结预训练模型的某些部分。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;要冻结模型的某些部分，只需将 .requires_grad_(False) 应用于不希望更新的参数。正如上面所述，因为使用这些参数作为输入的计算在前向传播中不会被记录，所以它们在反向传播中的 .grad 字段也不会被更新，因为它们一开始就不是反向图的一部分，正如期望的那样。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由于这是一个常见的模式，requires_grad 也可以在**模块级别使用 nn.Module.requires_grad_()** 进行设置。当应用于模块时，.requires_grad_() 对模块的所有参数（默认情况下 requires_grad=True）都起作用。<br>

## 1.6 梯度模式
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;除了设置 requires_grad 外，还有三种梯度模式可以从 Python 中选择，这些模式可以影响 PyTorch 中 autograd 在内部处理计算的方式：默认模式（grad 模式）、无梯度模式(no-grad mode)和推断模式(inference mode,)，所有这些模式都可以通过上下文管理器和装饰器进行切换。<br>

![figure1](images/autograd-figure1.jpg)

### 1.6.1 默认模式（Grad mode）
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**默认模式**是在没有启用其他模式（如无梯度模式和推断模式）时隐式存在的模式。与“无梯度模式”相对应的是，“默认模式”有时也被称为“grad 模式”。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;关于默认模式最重要的是，它是唯一可以生效的 requires_grad 模式。在其他两种模式中，requires_grad 总是被覆盖为 False。<br>

### 1.6.2 无梯度模式
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在无梯度模式下，计算行为就好像**没有任何输入需要梯度**一样。换句话说，在无梯度模式下，即使存在 require_grad=True 的输入，计算也不会被记录在反向图中。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当您需要执行不应由自动求导记录的操作，但仍希望在后续的梯度模式中使用这些计算的输出时，可以启用无梯度模式。这个上下文管理器使得在不必临时将张量设置为 requires_grad=False，然后再设置为 True 的情况下，方便地在一段代码或函数中禁用梯度。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;例如，**当编写优化器时，无梯度模式可能非常有用**：在执行训练更新时，您希望原地更新参数，而不会被自动求导记录。您还打算在下一次前向传播中使用**更新后的参数**进行梯度模式的计算。<br>

torch.nn.init 中的实现也依赖于无梯度模式，在初始化参数时避免了在原地更新初始化的参数时的自动求导跟踪。<br>

### 1.6.3 推断模式(inference mode)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;推断模式是无梯度模式的极端版本。与无梯度模式类似，推断模式下的计算不会被记录在反向图中，但启用推断模式将使得 PyTorch 加速您的模型。这种更好的运行时性能伴随着一个缺点：在退出推断模式后，无法在由自动求导记录的计算中使用在推断模式下创建的张量。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当进行不需要在反向图中记录的计算，并且您不打算在后续的自动求导记录的计算中使用在推断模式下创建的张量时，请启用推断模式。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;建议您在不需要自动求导跟踪的代码部分（例如数据处理和模型评估）中尝试推断模式。如果它能够适用于您的用例，那么可以免费获得性能提升。如果在启用推断模式后遇到错误，请检查您是否在退出推断模式后的自动求导记录的计算中使用了在推断模式下创建的张量。如果您无法避免在您的情况下使用这样的张量，您可以随时切换回无梯度模式。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;有关推断模式的详细信息，请参阅推断模式 [Inference Mode](https://pytorch.org/cppdocs/notes/inference_mode.html) 。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;有关推断模式的实现细节，请参阅 [RFC-0011-InferenceMode](https://github.com/pytorch/rfcs/pull/17) 。<br>

### 1.6.4 评估模式（nn.Module.eval()）
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;评估模式**不是一种局部禁用梯度计算的机制**。尽管如此，我们在这里还是包括了它，因为有时候它会被**误解**为这样的机制。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;从功能上讲，module.eval()（或等效地使用 module.train(False)）与无梯度模式和推断模式**完全无关**。model.eval() 对您的模型的影响完全取决于您模型中使用的具体模块以及它们是否定义了任何训练模式特定的行为。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果您的模型依赖于诸如 torch.nn.Dropout 和 torch.nn.BatchNorm2d 等模块，而这些模块在训练模式下可能表现不同，那么您需要负责调用 model.eval() 和 model.train()，例如在验证数据上避免更新 BatchNorm 的运行统计信息。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;建议您在训练模型时始终使用 model.train()，在评估模型（验证/测试）时使用 model.eval()，即使您不确定您的模型是否具有训练模式特定的行为，因为您使用的某个模块可能会在训练和评估模式下表现不同。<br>

## 1.7 In-place operations with autograd
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用自动求导（autograd）支持原地操作（In-place operations）是一个很复杂的问题，在大多数情况下，我们不鼓励使用它们。自动求导的缓冲区释放和重用使其非常高效，很少有情况下原地操作能够显著降低内存使用量。除非您的内存压力非常大，否则您可能永远不需要使用它们。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;存在两个主要原因限制了In-Place操作的适用性：<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;原地操作可能会**覆盖**计算梯度所需的值。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;每个原地操作都需要**重新构建计算图**。而不原地操作只是分配新的对象并保留对旧计算图的引用，而原地操作需要更改表示该操作的函数的所有输入的创建者。这可能会很棘手，特别是如果有许多张量引用相同的存储（例如通过索引或转置创建），并且如果被修改的输入的存储被其他张量引用，原地函数将引发错误。<br>





