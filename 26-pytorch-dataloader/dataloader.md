# 1 Dataset
- [pytorch link](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataset.py)

# 2 定义自己的数据集
```python
# 定义数据集
class MyDataset(data.Dataset):
    def __init__(self):
        # fake data
        self.data = torch.randn(100, 10)
        self.target = torch.randint(0, 2, (100,))
        
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
    def __len__(self):
        return len(self.data)
```

# 