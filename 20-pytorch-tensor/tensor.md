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
class TensorBase(metaclass=_TensorMeta): # metaclass 参数用于指定一个元类（metaclass）来创建类对象
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

# 4 TensorBase 方法汇总
- [TensorBase 链接](https://github.com/pytorch/pytorch/blob/main/torch/_C/__init__.pyi.in)
- [c++ api](https://pytorch.org/cppdocs/notes/tensor_basics.html)
- [c++ functions](https://pytorch.org/cppdocs/api/file_build_aten_src_ATen_Functions.h.html#file-build-aten-src-aten-functions-h)

# 4.1 魔术方法（基本运算符 + 构造函数 + 索引）
```python
def __abs__(self) -> Tensor: ... # 取绝对值
def __add__(self, other: Any) -> Tensor: ...
@overload
def __and__(self, other: Tensor) -> Tensor: ...
@overload
def __and__(self, other: Union[Number, _complex]) -> Tensor: ...
@overload
def __and__(self, other: Any) -> Tensor: ...
def __bool__(self) -> builtins.bool: ...
def __complex__(self) -> builtins.complex: ...
def __div__(self, other: Any) -> Tensor: ...
def __eq__(self, other: Any) -> Tensor: ... \# type: ignore[override]
def __float__(self) -> builtins.float: ... # # 调用 float() 函数时触发，将Tensor 数据类型转化为float
def __floordiv__(self, other: Any) -> Tensor: ...
def __ge__(self, other: Any) -> Tensor: ...
def __getitem__(self, indices: Union[None,_int, slice, Tensor, List, Tuple]) -> Tensor: ... # 用于通过索引或切片操作访问 Tensor 对象的元素或子集
def __gt__(self, other: Any) -> Tensor: ... # 
def __iadd__(self, other: Any) -> Tensor: ...
@overload
def __iand__(self, other: Tensor) -> Tensor: ...
@overload
def __iand__(self, other: Union[Number, _complex]) -> Tensor: ...
@overload
def __iand__(self, other: Any) -> Tensor: ...
def __idiv__(self, other: Any) -> Tensor: ...
def __ifloordiv__(self, other: Any) -> Tensor: ...
@overload
def __ilshift__(self, other: Tensor) -> Tensor: ... # 原地左移操作（<<=），移动位数有other 确定
@overload
def __ilshift__(self, other: Union[Number,_complex]) -> Tensor: ...
@overload
def __ilshift__(self, other: Any) -> Tensor: ...
def __imod__(self, other: Any) -> Tensor: ... # 原地取余操作
def __imul__(self, other: Any) -> Tensor: ...
def __index__(self) -> builtins.int: ... # 将 Tensor 对象用作整数索引，index = x[torch.tensor(2)] 时调用
@overload
def __init__(self, *args: Any, device: Device = None) -> None: ...
@overload
def __init__(self, storage: Storage) -> None: ...
@overload
def __init__(self, other: Tensor) -> None: ...
@overload
def __init__(self, size:_size, *, device: Device = None) -> None: ...
def __int__(self) -> builtins.int: ... # 转为整型 int() 方法调用
def __invert__(self) -> Tensor: ... # 按位取反操作符，（~）时被调用
@overload
def __ior__(self, other: Tensor) -> Tensor: ...
@overload
def __ior__(self, other: Union[Number, _complex]) -> Tensor: ...
@overload
def __ior__(self, other: Any) -> Tensor: ...
@overload
def __irshift__(self, other: Tensor) -> Tensor: ...
@overload
def __irshift__(self, other: Union[Number, _complex]) -> Tensor: ...
@overload
def __irshift__(self, other: Any) -> Tensor: ...
def __isub__(self, other: Any) -> Tensor: ...
@overload
def __ixor__(self, other: Tensor) -> Tensor: ...
@overload
def __ixor__(self, other: Union[Number, _complex]) -> Tensor: ...
@overload
def __ixor__(self, other: Any) -> Tensor: ...
def __le__(self, other: Any) -> Tensor: ...
def __long__(self) -> builtins.int: ...
@overload
def __lshift__(self, other: Tensor) -> Tensor: ...
@overload
def __lshift__(self, other: Union[Number, _complex]) -> Tensor: ...
@overload
def __lshift__(self, other: Any) -> Tensor: ...
def __lt__(self, other: Any) -> Tensor: ...
def __matmul__(self, other: Any) -> Tensor: ...
def __mod__(self, other: Any) -> Tensor: ...
def __mul__(self, other: Any) -> Tensor: ...
def __ne__(self, other: Any) -> Tensor: ... \# type: ignore[override]
def __neg__(self)  ->  Tensor: ...
def __nonzero__(self) -> builtins.bool: ... # 对 Tensor 对象进行真值测试（例如，用作条件表达式的条件）时触发
@overload
def __or__(self, other: Tensor) -> Tensor: ...
@overload
def __or__(self, other: Union[Number, _complex]) -> Tensor: ...
@overload
def __or__(self, other: Any) -> Tensor: ...
def __pow__(self, other: Any) -> Tensor: ...
def __radd__(self, other: Any) -> Tensor: ...
def __rand__(self, other: Any) -> Tensor: ...
def __rfloordiv__(self, other: Any) -> Tensor: ...
def __rmul__(self, other: Any) -> Tensor: ...
def __ror__(self, other: Any) -> Tensor: ...
def __rpow__(self, other: Any) -> Tensor: ...
@overload
def __rshift__(self, other: Tensor) -> Tensor: ...
@overload
def __rshift__(self, other: Union[Number, _complex]) -> Tensor: ...
@overload
def __rshift__(self, other: Any) -> Tensor: ...
def __rsub__(self, other: Any) -> Tensor: ...
def __rtruediv__(self, other: Any) -> Tensor: ...
def __rxor__(self, other: Any) -> Tensor: ...
def __setitem__(self, indices: Union[None,_int, slice, Tensor, List, Tuple], val: Union[Tensor, Number]) -> None: ... # 对 Tensor 对象进行赋值操作时触发
def __sub__(self, other: Any) -> Tensor: ...
def __truediv__(self, other: Any) -> Tensor: ... # 对 Tensor 对象使用除法操作符（/）时触发
@overload
def __xor__(self, other: Tensor) -> Tensor: ... # 
@overload
def __xor__(self, other: Union[Number, _complex]) -> Tensor: 
@overload
def __xor__(self, other: Any) -> Tensor: ...
```

## 4.2 私有方法
```python
# 执行矩阵乘法和加法操作，并应用激活函数。
# 首先：执行矩阵乘法操作：result = torch.matmul(mat1, mat2)；
# 然后：它将结果使用 alpha 权重进行缩放 与 input 张量相乘，其中input使用 beta 权重进行缩放；
# 最后：它应用激活函数（例如，Gelu）来对最终结果进行非线性变换，并返回变换后的张量。
def _addmm_activation(self, mat1: Tensor, mat2: Tensor, *, beta: Union[Number, _complex] = 1, alpha: Union[Number, _complex] = 1, use_gelu: _bool = False) -> Tensor: ...
def _autocast_to_full_precision(self, cuda_enabled:_bool, cpu_enabled:_bool) -> Tensor: ...# 自动转换到全精度
def _coalesced_(self, coalesced:_bool) -> Tensor: ... # 稀疏张量 sparse_tensor = torch.sparse_coo_tensor(indices, values, size) 中indices 可能有重复值，该函数用于在稀疏张量中合并重复的索引和值。
def _conj(self) -> Tensor: ... # 用于计算复数张量的共轭张量
def _conj_physical(self) -> Tensor: ... # tensor.conj().clone()
def _dimI(self)  ->  _int: ... # 稀疏矩阵 indices的维度
def _dimV(self)  >  _ int: ... # 稀疏矩阵 values 中每个 ele 的维度
def _indices(self) -> Tensor: ... # 获取稀疏矩阵的索引矩阵
def _is_all_true(self) -> Tensor: ... # bool 类型的Tensor 元素是否全是True
def _is_any_true(self) -> Tensor: ... # bool 类型的Tensor 元素是否含有True
def _is_view(self) ->_bool: ... # 检查张量是否是视图
def _is_zerotensor(self) ->_bool: ... # 检查一个Tensor 是否全为 0

# 用于创建子类时使用
@staticmethod
def _make_subclass(cls: Type[S], data: Tensor, require_grad: _bool = False, dispatch_strides: _bool = False, dispatch_device: _bool = False, device_for_backend_keys: Optional[_device] = None) -> S: ...
def _neg_view(self) -> Tensor: ... # 返回Tensor 中元素的相反数视图
def _nested_tensor_size(self) -> Tensor: ... # 嵌套张量(nested tensor)的PyTorch API处于原型阶段，并将在不久的将来进行更改。
def _nested_tensor_storage_offsets(self) -> Tensor: ... # 嵌套Tensor 的storage 偏移
def _nested_tensor_strides(self) -> Tensor: ... # 嵌套Tensor 的stride
def _nnz(self) ->_int: ... # non-zero elements 获取稀疏矩阵中非0元素的 个数
def _sparse_mask_projection(self, mask: Tensor, accumulate_matches: _bool = False) -> Tensor: ... # 获取经过掩模后的矩阵(self, mask, output 都是sparse的)
def _to_dense(self, dtype: Optional[_dtype] = None, masked_grad: Optional[_bool] = None) -> Tensor: ... # 转为 dense Tensor
@overload
def _to_sparse(self, *, layout: Optional[_layout] = None, blocksize: Optional[Union[_int, _size]] = None, dense_dim: Optional[_int] = None) -> Tensor: ... # 转为 sparse Tensor 
@overload
def _to_sparse(self, sparse_dim:_int) -> Tensor: ... # 转为sparse Tensor 默认为 ：Coordinate format
def _to_sparse_bsc(self, blocksize: Union[_int,__size], dense_dim: Optional[_int] = None) -> Tensor: ... # 转为块稀疏列布局
def _to_sparse_bsr(self, blocksize: Union[_int,__size], dense_dim: Optional[_int] = None) -> Tensor: ... # 转为块稀疏行布局
def _to_sparse_csc(self, dense_dim: Optional[_int] = None) -> Tensor: ... # 转为压缩稀疏列布局
def _to_sparse_csr(self, dense_dim: Optional[_int] = None) -> Tensor: ... # 转为压缩稀疏行布局
def _values(self) -> Tensor: ... # 返回稀疏Tensor 的value 矩阵
```
** 注释** <br>
- [嵌套张量 nested tensor](https://pytorch.org/docs/stable/nested.html)
- nested tensor 介绍：
One application of NestedTensors is to express sequential data in various domains.  While the conventional approach is to pad variable length sequences, NestedTensor enables users to bypass padding(绕过padding 操作).  The API for calling operations on a nested tensor is no different from that of a regular torch. Tensor, which should allow seamless integration(无缝整合) with existing models, with the main difference being construction of the inputs（主要区别在于输入的构造）. <br>
- python demo:
```python
import torch
a = torch.randn(50, 128) # text 1
b = torch.randn(32, 128) # text 2
nt = torch.nested.nested_tensor([a, b], dtype=torch.float32)
```
- [sparse tensor](https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs)

# 4.3 Tensor 的 对外API接口
```python
def abs(self) -> Tensor: ...
def abs_(self) -> Tensor: ... # 原地取绝对值
def absolute(self) -> Tensor: ... # abs 的别名
def absolute_(self) -> Tensor: ... # abs_ 的别名
def  acos(self)  ->  Tensor: ... # 反余弦
def acos_(self) -> Tensor: ...
def acosh(self) -> Tensor: ... # 逆双曲余弦
def acosh_(self) -> Tensor: ...
def add_(self, other: Union[Tensor, Number, torch.SymInt, torch.SymFloat], *, alpha: Optional[Number] = 1) -> Tensor: ...
def addcdiv(self, tensor1: Tensor, tensor2: Tensor, *, value: Union[Number, _complex] = 1) -> Tensor: ... # self + tensor1 / tensor2
def addcdiv_(self, tensor1: Tensor, tensor2: Tensor, *, value: Union[Number, _complex] = 1) -> Tensor: ...
def addcmul(self, tensor1: Tensor, tensor2: Tensor, *, value: Union[Number, _complex] = 1) -> Tensor: ...
def addcmul_(self, tensor1: Tensor, tensor2: Tensor, *, value: Union[Number, _complex] = 1) -> Tensor: ...
def adjoint(self) -> Tensor: ... # 返回共轭张量的视图，并将最后两个维度进行转置。
def align_as(self, other: Tensor) -> Tensor: ... # 按照other 中维度 来排列 self 的维度
@overload
def align_to(self, order: Sequence[Union[str, ellipsis, None]], ellipsis_idx: _int) -> Tensor: ...
@overload
def align_to(self, names: Sequence[Union[str, ellipsis, None]]) -> Tensor: ...
@overload
def all(self) -> Tensor: ... # 全为真则为真
@overload
def all(self, dim: _int, keepdim: _bool = False) -> Tensor: ...
@overload
def all(self, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> Tensor: ...
def allclose(self, other: Tensor, rtol: _float = 1e-05, atol: _float = 1e-08, equal_nan: _bool = False) -> bool: ... # 两数据差异比较 ： ∣input−other∣ ≤ atol + rtol × ∣other∣
def amin(self, dim: Union[_int,_size] = (), keepdim:_bool = False) -> Tensor: ... # 返回输入张量在给定维度 dim 中的每个切片的最小值
def aminmax(self, *, dim: Optional[_int] = None, keepdim: _bool = False) -> torch.return_types.aminmax: ... # 返回输入张量在给定维度 dim 中的每个切片的最小值 和 最小值 两个Tensor
def angle(self) -> Tensor: ...
@overload
def any(self) -> Tensor: ...
@overload
def any(self, dim:_int, keepdim: _bool = False) -> Tensor: ...
@overload
def any(self, dim: Union[str, ellipsis, None], keepdim: _bool = False) -> Tensor: ...
def apply_(self, callable: Callable) -> Tensor: ...
def arccos(self) -> Tensor: ...
def arccos_(self) -> Tensor: ...
def arccosh(self) -> Tensor: ...
def arccosh_(self) -> Tensor: ...
def arcsin(self) -> Tensor: ...
def arcsin_(self) -> Tensor: ...
def arcsinh(self) -> Tensor: ...
def arcsinh_(self) -> Tensor: ...
def arctan(self) -> Tensor: ...
def arctan2(self, other: Tensor) -> Tensor: ...
def arctan2_(self, other: Tensor) -> Tensor: ...
def arctan_(self) -> Tensor: ...
def arctanh(self) -> Tensor: ...
def arctanh_(self) -> Tensor: ... # 原地操作
def argmax(self, dim: Optional[_int] = None, keepdim: _bool = False) -> Tensor: ... # 返回最大值的索引(拉平后)
def argmin(self, dim: Optional[_int] = None, keepdim: _bool = False) -> Tensor: ...
@overload
def argsort(self, *, stable: _bool, dim: _int = -1, descending: _bool = False) -> Tensor: ... # 返回Tensor 排序后的 index 组成的Tensor
@overload
def argsort(self, dim: _int = -1, descending: _bool = False) -> Tensor: ... # torch.sort 是用来排序的
@overload
def argsort(self, dim: Union[str, ellipsis, None], descending: _bool = False) -> Tensor: ...
def argwhere(self) -> Tensor: ...
def as_strided(self, size: Sequence[Union[_int, SymInt]], stride: Sequence[Union[_int, SymInt]], storage_offset: Optional[Union[_int, SymInt]] = None) -> Tensor: ...
def as_strided_(self, size: Sequence[Union[_int, SymInt]], stride: Sequence[Union[_int, SymInt]], storage_offset: Optional[Union[_int, SymInt]] = None) -> Tensor: ...
def as_strided_scatter(self, src: Tensor, size: Sequence[Union[_int, SymInt]], stride: Sequence[Union[_int, SymInt]], storage_offset: Optional[Union[_int, SymInt]] = None) -> Tensor: 
def as_subclass(self, cls: Type[S]) -> S: ...
def asin(self) -> Tensor: ...
```

