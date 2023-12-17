import torch
import torch.nn as nn

def conv_demo():
    '''
    作业：
    实现 1x1 卷积、组卷积、膨胀卷积、深度可分离卷积
    '''
    # With square kernels and equal stride
    m = nn.Conv2d(16, 33, 3, stride=2) # 实例化一个conv2d
    # non-square kernels and unequal stride and with padding
    # m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    # non-square kernels and unequal stride and with padding and dilation
    # m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    input = torch.randn(20, 16, 50, 100) # activation
    output = m(input) # 传入一个数据，让torch 来计算
    print(output)
    
def transposed_conv_demo():
    # With square kernels and equal stride
    m = nn.ConvTranspose2d(16, 33, 3, stride=3)
    # non-square kernels and unequal stride and with padding
    # m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    input = torch.randn(20, 16, 50, 100)
    output = m(input)
    print(output)
    # # exact output size can be also specified as an argument
    # input = torch.randn(1, 16, 12, 12)
    # downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
    # upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
    # h = downsample(input)
    # h.size()

    # output = upsample(h, output_size=input.size())
    # output.size()

def demo_linear():
    m = nn.Linear(20, 30) # A[m,k] * B[k, n] = O[m, n] --> k = 20(input_features), n=30(out_features)
    input = torch.randn(128, 20)
    output = m(input)
    print(output.size())

def matmul_demo():
    tensor1 = torch.randn(10, 2, 3, 4)
    tensor2 = torch.randn( 1, 2, 4, 5)
    output = torch.matmul(tensor1, tensor2) # 支持多维矩阵相乘
    print(output.shape)

    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 3)
    output = torch.mm(mat1, mat2) # 只支持二维矩阵
    print(output.shape)

    input = torch.randn(10, 3, 4)
    mat2 = torch.randn(10, 4, 5)
    res = torch.bmm(input, mat2) # 只支持三维的矩阵相乘
    res.size()
    
def bn_demo():
    # With Learnable Parameters
    m = nn.BatchNorm2d(100) # 100 --> channel
    # Without Learnable Parameters
    # m = nn.BatchNorm2d(100, affine=False)
    input = torch.randn(20, 100, 35, 45)
    output = m(input)
    
def ln_demo():
    batch, sentence_length, embedding_dim = 20, 5, 10 # 
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    layer_norm = nn.LayerNorm(embedding_dim)
    # Activate module
    output = layer_norm(embedding)
    print(output)
    
def cat_stack_demo():
    x = torch.randn(2, 3)
    output = torch.cat((x, x, x), 0)
    output2 = torch.cat((x, x, x), 1)
    
    a = torch.randn(2,3)
    b = torch.randn(2,3)
    c= torch.stack([a,b], dim=1)
    
    x = torch.tensor([[1], [2], [3]])
    x.size()
    x.expand(3, 4)
    x.expand(-1, 4)   # -1 means not changing the size of that dimension

def split_demo():
    a = torch.arange(10).reshape(5, 2)
    output = torch.split(a, 2)
    output2 = torch.split(a, [1, 4])
    print(a)

def embedding_demo():
    embedding = nn.Embedding(30522, 768, padding_idx=0)
    input = torch.LongTensor([[0, 2, 0, 5], [3, 5, 6, 20]])
    output = embedding(input)
    print(output.shape)

if __name__ == "__main__":
    # conv_demo()
    # transposed_conv_demo()
    # demo_linear()
    # matmul_demo()
    # bn_demo()
    # ln_demo()
    # cat_stack_demo()
    # split_demo()
    embedding_demo()
    print("run op_test.py successfully !!!")