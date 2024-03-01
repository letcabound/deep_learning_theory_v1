# 1 案例展示
```shell
# install pytorch > 1.12 with cuda enabled first

git clone https://github.com/pytorch/examples.git (2d725b6ab255e05c55e0b08925f06f171aaedc0c)
cd examples/distributed/FSDP
pip install -r requirements.txt
torchrun --nnodes 1 --nproc_per_node 2  T5_training.py

```

# 2 原理及流程
- DDP 实现原理 <br>
![DDP 原理](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Graph-2.png)

- FSDP 实现原理 <br>
![FSDP 原理](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-Graph-2.png)

- DDP 和 FSDP 通信上的不同
![通信](https://engineering.fb.com/wp-content/uploads/2021/07/FSDP-graph-2a.png)

# 3 注意事项
- 优化器必须在模块被封装之后初始化，因为FSDP会就地分片参数，这会破坏之前初始化的优化器;
- 在使用CPU卸载时，FSDP目前不支持在 "no_sync()" 之外进行梯度累积。尝试这样做会产生错误的结果，因为FSDP将使用新的已 reduced 的梯度，而不是与任何现有梯度累积;
- 模型构造之后如何更改 parameter的名字 将导致未定义的行为;
- 传递sync_module_states=True标志要求将模块(module)放置在GPU上，或者使用device_id参数来指定FSDP将模块移动到的CUDA设备。这是因为sync_module_states=True需要进行GPU通信;
- 截至PyTorch 1.12，FSDP仅提供有限的共享参数支持（例如，将一个Linear层的权重设置为另一个层的权重）。特别是，共享参数的模块必须作为同一FSDP单元的一部分进行封装。如果您的使用情况需要增强的共享参数支持;
- 尝试运行包含在FSDP实例中的子模块的前向传递是不受支持的，并且会导致错误。这是因为子模块的参数将被分片，但它本身不是一个FSDP实例，所以它的前向传递不会适当地聚集所有参数。这可能会在尝试仅运行编码器的编码器-解码器模型时发生，并且编码器没有被封装在自己的FSDP实例中。为了解决这个问题，请将子模块封装在自己的FSDP单元中。;
- 在运行FSDP的“forward”函数之前，输入将被移动到计算设备（与FSDP模块所在的设备相同），因此用户无需手动将输入从CPU移动到GPU。

# 4 核心代码解读
## 4.1 FSDP 初始化
```python
# FSDP 的实例化
model = torch.distributed.fsdp.FullyShardedDataParallel(model,
    auto_wrap_policy=t5_auto_wrap_policy,
    mixed_precision=mixed_precision_policy,
    sharding_strategy=fsdp_config.sharding_strategy,
    device_id=torch.cuda.current_device(),
    limit_all_gathers=fsdp_config.limit_all_gathers)

# 具体类的构造函数
class FullyShardedDataParallel(nn.Module, _FSDPState):
    def __init__(
      self,
      module: nn.Module, # 将被FSDP 包装的module
      # 用于进行集体通信的进程组，也是模型分片的进程组。对于混合分片策略（例如ShardingStrategy.HYBRID_SHARD），用户可以传入一个包含表示要进行分片和复制的进程组的元组。
      process_group: ProcessGroupType = None, 
      sharding_strategy: Optional[ShardingStrategy] = None, #配置FSDP使用的分片策略，在内存节省和通信开销之间做出权衡
      cpu_offload: Optional[CPUOffload] = None, #是否启动CPU Offload 默认 None
      auto_wrap_policy: Optional[Union[Callable, _FSDPPolicy]] = None,
      backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE, # 明确配置 all-gather 操作的反向预取。
      mixed_precision: Optional[MixedPrecision] = None, # 为FSDP配置本地混合精度。可以设置参数、缓冲区和梯度reduce的数据类型。
      ignored_modules: Optional[Iterable[torch.nn.Module]] = None, # 不进行分片的模块
      param_init_fn: Optional[Callable[[nn.Module], None]] = None, # 
      device_id: Optional[Union[int, torch.device]] = None,
      sync_module_states: bool = False,
      forward_prefetch: bool = False,
      limit_all_gathers: bool = False,
      use_orig_params: bool = False,
      ignored_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
  ):


# auto_wrap_policy (Optional[Union[Callable[[nn.Module, bool, int], bool], _FSDPPolicy]]):<br>
# 这可以是None、一个_FSDPPolicy，或者一个具有固定签名的可调用对象。如果是None，则module将只被一个顶层的FSDP实例包装，没有任何嵌套包装。
# 如果是一个_FSDPPolicy，则包装遵循给定的策略。torch.distributed.fsdp.wrap.py中的ModuleWrapPolicy是一个示例。
# 如果是一个可调用对象，则它应该接受三个参数module: nn.Module、recurse: bool和nonwrapped_numel: int，并返回一个bool值，指定当recurse=False时是否应该包装传入的module，或者如果recurse=True时遍历应该继续下去。
# 可调用对象可以添加额外的自定义参数。torch.distributed.fsdp.wrap.py中的size_based_auto_wrap_policy是一个示例可调用对象，它在其子树中的参数超过100M个元素时包装一个模块。一个好的做法是在包装后打印模型，并根据需要进行调整。

# ignored_modules (Optional[Iterable[torch.nn.Module]])
# 被当前实例忽略的模块, 模块的参数以及子模块的参数和缓冲区将被忽略。
# ignored_modules中的直接模块不应该是:class:FullyShardedDataParallel的实例，如果已经构建的子模块是:class:FullyShardedDataParallel的实例并且嵌套在这个实例下面，那么它们将不会被忽略。
# 当使用auto_wrap_policy时，或者如果参数的分片不由FSDP管理，可以使用此参数来避免按模块粒度分片特定的参数。

# param_init_fn（Optional[Callable[[nn.Module], None]]）：
# 一个 Callable[torch.nn.Module] -> None，用于指定当前在元设备上的模块应该如何初始化到实际设备上.
# 请注意，从v1.12开始，我们通过is_meta检查来检测元设备上的模块，并应用默认初始化，如果未指定param_init_fn, 则在传入的nn.Module上调用reset_parameters方法，否则我们运行param_init_fn来初始化传入的nn.Module。
# 特别要注意的是，如果任何需要被FSDP包装的模块的 Parameters 的is_meta=True且未指定param_init_fn，我们假设您的模块正确实现了reset_parameters()，否则会抛出错误。
# 此外，我们还支持使用torchdistX（https://github.com/pytorch/torchdistX）的``deferred_init`` API初始化的模块。
# 在这种情况下，延迟初始化的模块将通过调用torchdistX的materialize_module或传入的param_init_fn（如果不为None）来进行默认初始化。同一个Callable被应用于初始化所有的元模块。请注意，在进行任何FSDP分片逻辑之前，将应用此初始化函数。

```

**参数解析:** <br>
- 


# 5 参考文档
- [Pytorch FSDP Paper](https://arxiv.org/pdf/2304.11277.pdf)
- [Engineering at meta](https://engineering.fb.com/2021/07/15/open-source/fsdp/)
- [Pytorch Doc](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)

