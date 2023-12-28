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
    nbytes: _int
    itemsize: _int
    _has_symbolic_sizes_strides: _bool
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


