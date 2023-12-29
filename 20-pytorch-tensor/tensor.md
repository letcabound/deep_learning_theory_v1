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
def__reduce_ex_internal(self, proto): '. # 被__reduce_ex__调用
def __setstate__(self, state):  # 自定义在反序列化过程中恢复对象状态的行为
def __repr__(self, *, tensor_contents=None): # 定义对象的字符串表示形式。当调用 repr(obj) 或在交互式环境中直接输入 obj 时，Python 会调用对象的 __repr__ 方法来获取对象的字符串表示。
def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None): # 计算当前张量相对于计算图叶节子节点的梯度
def register_hook(self, hook): # 注册反向钩子函数，计算完该张量的梯度时，钩子（hook）将被调用。
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
def resize(self,~*sizes): # 
def resize_as(self, tensor): '..
def split(self, split_size,  \operatorname{dim}=0): \cdots 
def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):  \cdots 
def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):  \cdots 
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def__rsub_(self, other): '.
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def__rdiv_(self, other): '.

_rtruediv__  =  _rdiv_
_itruediv_  =  _C._TensorBase._idiv_
_ pow_ = _handle_torch_function_and_wrap_type_error_to_not_implemented (  \cdots 
ipow_ = _handle_torch_function_and_wrap_type_error_to_not_implemented (  \cdots 
@_handle_torch_function_and_wrap_type_error_to_not_implemented def__rmod_(self, other): '.
def_format_(self, format_spec):
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def__rpow_(self, other):
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def_floordiv_(self, other):
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def_rfloordiv_(self, other):
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def__rlshift_(self, other):
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def_rrshift_(self, other):  \cdots 
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def__rmatmul_(self, other):


__pos__ = _C._TensorBase.positive 
__neg_ = _C._TensorBase.neg 
__abs__ = _C._TensorBase.abs

def__len_(self):  \cdots 
def__iter_(self): '.
def__hash_(self): '.
def__dir_(self):  \cdots 
\# Numpy array interface, to support "numpy.as
_array_priority_  =1000  \# prefer Tensor op
def__array_(self, dtype=None): '..
\# Wrap Numpy array again in a suitable tensor
\# 'numpy.sin(tensor) -> tensor" or 'numpy.gre
def__array_wrap_(self, array): ...
def__contains_(self, element): '.
@property
def__cuda_array_interface_(self):  \cdots 
def storage_type(self): '..
def refine_names(self, *names): '.
def align_to(self, *names): ...
def unflatten(self, dim, sizes):  \cdots 
def rename_(self, *names, **rename_map):  \cdots 
def rename(self, *names, **rename_map):
def to_sparse_coo(self):  \cdots

def_update_names(self, names, inplace):  \cdots 
@classmethod
def_torch_function_(cls, func, types, args=(), kwargs=None):
__torch_dispatch__ = C._disabled_torch_dispatch_impl
def__dlpack_(self, stream=None): '..
def__dlpack_device_(self) -> Tuple[enum.IntEnum, int]: ..
__module__ = "torch"

```

