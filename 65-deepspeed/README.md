# run
```python
pip install deepspeed
git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/training/cifar
bash run_ds.sh
```

# 相关环境依赖
- gcc 版本要小于 10;
- /usr/bin/gcc --> 链接到 /usr/bin/gcc-7 即可;

# 启动指令
```python
deepspeed --bind_cores_to_rank cifar10_deepspeed.py --deepspeed $@
```
其中: <br>
-  deepspeed 为可执行脚本：miniconda3/envs/pytorch2.0/bin/deepspeed
```python
#!/home/mtn/miniconda3/envs/pytorch2.0/bin/python
# EASY-INSTALL-DEV-SCRIPT: 'deepspeed==0.14.1+ffb53c25','deepspeed'
__requires__ = 'deepspeed==0.14.1+ffb53c25'
__import__('pkg_resources').require('deepspeed==0.14.1+ffb53c25')
__file__ = '/home/mtn/DeepSpeed/bin/deepspeed'
with open(__file__) as f:
    exec(compile(f.read(), __file__, 'exec'))
```

- 启动的文件为 /home/mtn/DeepSpeed/bin/deepspeed：
```python
#!/usr/bin/env python3

from deepspeed.launcher.runner import main

if __name__ == '__main__':
    main()
```


