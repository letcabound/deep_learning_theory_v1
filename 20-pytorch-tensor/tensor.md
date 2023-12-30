# [pytorch tensor](https://pytorch.org/docs/stable/tensors.html)
A torch.Tensor is a multi-dimensional matrix containing elements of a single data type. <br>

# 1 创建pytorch tensor

## 1.1 用torch.Tensor 创建
```python
import torch
data = [[1, 2],[3, 4]] # python list
x_data = torch.tensor(data) # 
x_data2 = torch.tensor((1, 2, 3))
# x_data3 = torch.tensor({"a": 5}) # fail
print("x_data2: ", x_data2)
```

## 1.2 直接生成特殊的tensor
```python
import torch
data = torch.ones(1, 2, 3)
data1 = torch.zeros(1, 3,4)
data2 = torch.randn(3, 4, 5)
data3 = torch.eye(4, 5)
data4 = torch.randint(5, (2, 10))
print("data type: ", type(data4))
print("data2: ", data4)
```

## 1.3 仿照其它tensor生成
```python
import torch
data0 = torch.Tensor([1, 2, 3])
data1 = torch.ones_like(data0)
data2 = torch.empty_like(data0)
# data3 = torch.empty_like(data0)
print("data: ", data2)
```

## 1.4 从numpy生成
```python
  np_array = np.array([1, 2, 3])
  tensor_numpy = torch.from_numpy(np_array)
  # tensor_numpy2 = torch.Tensor(np_array) # deepcopy 了一份数据
  np_array[0] = 100
  data_numpy = tensor_numpy.numpy()
  # print("data numpy: ", type(data_numpy))
  print("numpy tensor: ", tensor_numpy)
```

# 2 Tensor 的属性
- [pytorch Tensor class](https://github.com/pytorch/pytorch/blob/main/torch/_tensor.py)
- [pytorch C TensorBase](https://github.com/pytorch/pytorch/blob/main/torch/_C/__init__.pyi.in)
- [官方文档](https://pytorch.org/docs/stable/tensors.html)

```python
# Defined in torch/csrc/autograd/python_variable.cpp
class TensorBase(metaclass=_TensorMeta):
    requires_grad: _bool 
    retains_grad: _bool
    shape: Size
    data: Tensor
    names: List[str]
    device: _device
    dtype: _dtype
    layout: _layout
    real: Tensor
    imag: Tensor
    T: Tensor # only 2d 
    H: Tensor # Returns a view of a matrix (2-D tensor) conjugated and transposed. （返回一个矩阵（2D 张量）的共轭转置视图）
    mT: Tensor # Returns a view of this tensor with the last two dimensions transposed.
    mH: Tensor
    ndim: _int
    output_nr: _int # 创建者的第几个输出, requres_grad = True 时生效
    _version: _int
    _base: Optional[Tensor]
    _cdata: _int
    grad_fn: Optional[_Node]
    _grad_fn: Any
    _grad: Optional[Tensor]
    grad: Optional[Tensor]
    _backward_hooks: Optional[Dict[_int, Callable[[Tensor], Optional[Tensor]]]]
    ${tensor_method_hints}

_TensorBase = TensorBase
```

- 注释1： 对于复数，共轭转置操作将实部保持不变，而虚部取负值。例如，对于复数 z = a + bi，其共轭转置为 z* = a - bi。<br>
- 注释2：对于矩阵，共轭转置将矩阵的每个元素取复共轭，然后对矩阵进行转置操作。<br>

```python
import torch
a = torch.randn(4,5,6)
a.requires_grad=True
aa = a.split([1,2,1])
print(aa[1].output_nr)
```


# 3 外层 Tensor 方法汇总
```python
def __deepcopy__(self, memo):   #自定义对象在深拷贝（deep copy）操作中的行为
def__reduce_ex__(self, proto):  #自定义对象在序列化和反序列化过程中的行为
def storage(self):  # 返回与张量关联的底层数据存储对象
def _typed_storage(self): # For internal use only, to avoid raising deprecation warning <new>.
def__reduce_ex_internal(self, proto): '. # 被__reduce_ex__调用
def __setstate__(self, state):  # 自定义在反序列化过程中恢复对象状态的行为
def __repr__(self, *, tensor_contents=None): # 定义对象的字符串表示形式。当调用 repr(obj) 或在交互式环境中直接输入 obj 时，Python 会调用对象的 __repr__ 方法来获取对象的字符串表示。
def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None): # 计算当前张量相对于计算图叶节子节点的梯度
def register_hook(self, hook): # 注册反向钩子函数，计算完该张量的梯度时，钩子（hook）将被调用。
def register_post_accumulate_grad_hook(self, hook): # 注册一个梯度累加之后的反向钩子函数 <new>.
def reinforce(self, reward): # 强制报错
detach  =  _C.__add_docstr(_C._TensorBase.detach）# Returns a new Tensor, detached from the current graph.
detach_  =  _C.__add_docstr(_C._TensorBase.detach_) # 将张量从创建它的计算图中分离，使其成为叶节点。
def is_shared(self): # 检查张量是否位于共享内存中。对于 CUDA 张量（即在 GPU 上的张量），该方法始终返回 True
def share_memory_(self): # 将底层存储移动到共享内存 Moves the underlying storage to shared memory
def__reversed__(self):  # 按照某个维度对张量进行反转操作，可以用 torch.flip() 调用
def norm(self, p="fro", dim=None, keepdim=False, dtype=None): # 参考torch.norm --> 返回给定张量的矩阵范数或向量范数。
def solve(self, other): # 解线性方程组，参考torch.solve() ： torch.solve(input, A) -> (solution, LU) 
def lstsq(self, other):  # 求解线性最小二乘问题
def eig(self, eigenvectors=False): # 计算一个实对称或复数方阵的特征值和特征向量。
def lu(self, pivot=True, get_infos=False): # 执行 LU 分解（LU decomposition） 
def stft(...) # 短时傅里叶变换
def istft (...）  短时傅里叶逆变换
def resize(self,~*sizes): # 根据需要重新分配内存，可能返回原始张量的视图（view）或副本（copy）
def resize_as(self, tensor): # 按照另一个tensor的 size 来 resize
def split(self, split_size,  dim=0): # 同torch.split，将一个tensor split 成多个tensor
def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None): # 返回输入张量的唯一元素，去重作用，输出为一维的
def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None): # 消除每个连续数据中除第一个元素之外的所有元素
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def__rsub__(self, other): # 定义tensor 右侧减法 result = other - self 的行为
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def__rdiv__(self, other): # 定义tensor 右侧除法的行为, 使用运算符 // 进行操作，表示执行整数除法或向下取整除法，目前已被 __rfloatdiv__  代替

__rtruediv__  =  __rdiv__ # 右侧真除法使用运算符 / 进行操作，表示执行浮点数除法
__itruediv_  =  _C._TensorBase._idiv_ # 用于实现就地真除法运算符 /= 的行为
__ pow__ = _handle_torch_function_and_wrap_type_error_to_not_implemented(_C._TensorBase.pow) # 实现幂运算，即通过运算符 ** 进行操作
__ipow__ = _handle_torch_function_and_wrap_type_error_to_not_implemented (_C._TensorBase.pow_) # 实现就地幂运算，即通过运算符 **= 进行操作

def__format__(self, format_spec): # 使用内置的 format() 函数 进行格式化时调用

@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rmod__(self, other): # 定义右侧取模运算符（取余运算符）% 的行为
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rpow__(self, other): # 定义右侧取幂运算符 ** 的行为
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __floordiv__(self, other): # 用于实现整数除法运算 // 的行为。
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rfloordiv__(self, other): # 实现右侧整数除法运算 // 的行为
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rlshift__(self, other): # 实现右侧的位左移运算符 << 的行为
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rrshift_(self, other):  # 实现右侧的位右移运算符 >> 的行为 
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rmatmul__(self, other): # 用于实现右侧矩阵乘法运算符 @ 的行为

__pos__ = _C._TensorBase.positive # 实现正号运算符（一元加法）的行为
__neg_ = _C._TensorBase.neg # 实现负号运算符（一元加法）的行为
__abs__ = _C._TensorBase.abs # 实现绝对值运算符（一元加法）的行为

def __len__(self):  # 返回 Tensor 对象的维度
def __iter__(self): # 用于使 Tensor 对象可迭代
def __hash__(self): # 支持 Tensor 对象的哈希操作。
def __dir__(self):  # 用于返回 Tensor 对象的属性和方法列表。
__array_priority_  =1000 # support `numpy.asarray(tensor) -> ndarray
def __array_(self, dtype=None): # 支持将 Tensor 对象转换为 NumPy 数组
def __array_wrap__(self, array): # 用np 操作Tensor对象，并返回 Tensor： numpy.sin(tensor) -> tensor
def __contains__(self, element): # 用于判断某个值是否存在于 Tensor 对象中，用 in 来触发 

@property
def __cuda_array_interface_(self): # CUDA 张量的数组视图描述
def storage_type(self): # 存储的数据类型
def refine_names(self, *names): # 修改或细化 Tensor 对象的维度命名 ：x = torch.randn(3, 4, 5, names=('batch', 'channel', 'height'))
def align_to(self, *names): # 调整 Tensor 对象的维度顺序以匹配指定的顺序： y = x.align_to('height', 'batch', 'channel')
def unflatten(self, dim, sizes): # 用于将一个连续的一维 Tensor 对象重新转换为具有指定维度大小的多维 Tensor 对象
def rename_(self, *names, **rename_map): # rename_ 是一个方法，用于原地修改 Tensor 对象的维度名称。
def rename(self, *names, **rename_map): # 修改 Tensor 对象的维度名称
def to_sparse_coo(self): # 用于将一个稠密（dense）的 Tensor 对象转换为 COO（Coordinate）格式的稀疏（sparse）Tensor 对象
def dim_order(self): 返回一个由整数组成的元组，描述了 self 的维度顺序或物理布局 <new>
def _update_names(self, names, inplace): # 原地更新 Tensor 对象的维度名称。
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None): # 对子类进行包装，使得在子类上调用的方法返回一个子类实例，而不是一个 torch.Tensor 实例， 需要自己在该方法内做转化
__torch_dispatch__ = C._disabled_torch_dispatch_impl # 用于定义自定义的分发逻辑，以便根据输入的类型、形状或其他条件来选择适当的实现。
def __dlpack__(self, stream=None): # 用于将 Tensor 对象转换为 DLPack 格式。DLPack 是一种用于在不同深度学习框架之间交换数据的标准化格式
def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]: # 用于获取 Tensor 对象所在的设备信息并返回。DLPack 是一种用于在不同深度学习框架之间交换数据的标准化格式
__module__ = "torch" # 用于指示 Tensor 对象所属的模块
```

