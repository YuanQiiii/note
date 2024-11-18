## 作业
### 01
### 02
```python
# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""

# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np

from tqdm import tqdm

import os

print("Current working directory:", os.getcwd())

x = 'C:/Users/exqin/Desktop/hw2'

# 用于解决工作路径不正确导致的file not found

os.chdir(x)

print("Current working directory:", os.getcwd())

# 加载数据集,numpy格式

# 训练集
X_train = np.load('.\mnist\X_train.npy') # (60000, 784), 数值在0.0~1.0之间

# 训练集特征数据，形状为 (60000, 784)，共有 60000 个样本，每个样本是一个长度为 784 的一维向量，数值范围在 0.0 到 1.0 之间。
y_train = np.load('.\mnist\y_train.npy') # (60000, )

# 训练集标签数据，形状为 (60000,)，共有 60000 个标签。
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码
# 原始标签 y_train 中的某个样本的取值为 i，那么经过 np.eye(10)[y_train] 转换后，这个样本的标签将变为一个长度为 10 的向量，第 i 个位置为 1，其余位置为 0。

# 验证集
X_val = np.load('.\mnist\X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('.\mnist\y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

# 测试集
X_test = np.load('.\mnist\X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('.\mnist\y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码


# 定义激活函数
def relu(x): # 当输入为一个矩阵的时候,函数会对每一个元素进行操作
    '''
    relu函数
    '''
    return np.maximum(0,x)


def relu_prime(x):
    '''
    relu函数的导数
    '''
    return np.where(x>0,1,0)


# 输出层激活函数
def f(x):
    '''
    softmax函数, 防止除0
    '''
    e_x=np.exp(x-np.max(x,axis=1,keepdims=True))
    return e_x / np.sum(e_x,axis=1,keepdims=True)



# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return  -np.sum(y_true * np.log(y_pred+1e-15),axis=-1)


def loss_fn_prime(y_true, y_pred):  # 注意这里是交叉熵损失函数和softmax函数的复合函数求导的结果
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    return (y_pred - y_true)/y_true.shape[0] # (batch_size,10)


# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0 / shape[0]), size=shape)


# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size, output_size, lr=0.10):
        '''
        初始化网络结构
        '''

        self.W1 = init_weights((input_size,hidden_size))
        self.b1 = init_weights((hidden_size,))

        self.W2 = init_weights((hidden_size,128))
        self.b2 = init_weights((128,))

        self.W3 = init_weights((128, 64))
        self.b3 = init_weights((64,))

        self.W4 = init_weights((64, output_size))
        self.b4 = init_weights((output_size,))

        # 记录前向传播的中间结果,用于更新梯度
        self.z1 = 0
        self.a1 = 0
        self.z2 = 0
        self.a2 = 0
        self.z3 = 0
        self.a3 = 0
        self.z4 = 0
        self.a4 = 0
        # 学习率
        self.lr = lr

    def forward(self, x): # (batch_size, 784)   
        '''
        前向传播
        '''
        self.z1 = np.dot(x, self.W1) + self.b1; # (batch_size,784) * (784,256) + (1,256) 这里b1被广播

        self.a1 = relu(self.z1) # (batch_size,256)

        self.z2 = np.dot(self.a1, self.W2) + self.b2 # (batch_size,256) * (256,128) + (1,128)

        self.a2 = relu(self.z2) # (batch_size,128)

        self.z3 = np.dot(self.a2, self.W3) + self.b3 # (batch_size,128) * (128,64) + (1,64)

        self.a3 = relu(self.z3) # (batch_size,64) 

        self.z4 = np.dot(self.a3, self.W4) + self.b4 # (batch_size,64) * (64,10) + (1,10)

        self.a4 = f(self.z4) # (batch_size,10)

        return self.a4 # (batch_size,10)


    def step(self, x_batch, y_batch): # (batch_size, 784)   (batch_size, 10)
        '''
        一步训练
        '''
        batch_loss = 0
        batch_acc = 0
        # 前向传播

        y_pred = self.forward(x_batch) # (batch_size, 10)

        # 计算损失和准确率
        batch_loss = np.mean(loss_fn(y_batch,y_pred)) # (batch_size,10) ==> (1,1)
        batch_acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1)) # 在对应行找出两者最大的位置的索引,看是否相等,相等即为acc  ==>  (1,1)

        # 反向传播

        # 梯度矩阵
        self.grads_W4 = np.zeros_like(self.W4)
        self.grads_b4 = np.zeros_like(self.b4)
        self.grads_W3 = np.zeros_like(self.W3)
        self.grads_b3 = np.zeros_like(self.b3)
        self.grads_W2 = np.zeros_like(self.W2)
        self.grads_b2 = np.zeros_like(self.b2)
        self.grads_W1 = np.zeros_like(self.W1)
        self.grads_b1 = np.zeros_like(self.b1)
        # 求导
        delta_l_z4 = loss_fn_prime(y_batch,y_pred)  # (batch_size, 10) 对线性模型输出值求导
        # print(delta_l_z4.shape)
        delta_l_a3 = np.dot(delta_l_z4 ,self.W4.T) # (batch_size, 10) * (10, 64) 后来的求导结果放在右边
        # print(delta_l_a3.shape)
        delta_l_z3 = relu_prime(self.z3)*delta_l_a3 # (batch_size, 64) * (10, 64) 
        # print(delta_l_z3.shape)
        delta_l_a2 = np.dot(delta_l_z3, self.W3.T) # (batch_size, 64) * (64, 128)
        # print(delta_l_a2.shape)
        delta_l_z2 = relu_prime(self.z2)*delta_l_a2 # (batch_size, 64) * (64, 128)
        # print(delta_l_z2.shape)
        delta_l_a1 = np.dot(delta_l_z2, self.W2.T)  # (batch_size, 128) * (128, 256) 
        # print(delta_l_a1.shape)
        delta_l_z1 = relu_prime(self.z1)*delta_l_a1 # (batch_size, 128) * (128, 256) 
        # print(delta_l_z1.shape)
        # 更新参数梯度
        self.grads_W4 = np.dot(self.a3.T,delta_l_z4)  # (64,batch_size) * (batch_size, 10)
        self.grads_b4 = np.sum(delta_l_z4, axis=0) # (10,)
        self.grads_W3 = np.dot(self.a2.T, delta_l_z3) # (128,batch_size) *(batch_size, 64)
        self.grads_b3 = np.sum(delta_l_z3, axis=0) # (64,)
        self.grads_W2 = np.dot(self.a1.T, delta_l_z2) # (256,batch_size) *(batch_size, 128)
        self.grads_b2 = np.sum(delta_l_z2, axis=0) # (128,)
        self.grads_W1 = np.dot(x_batch.T, delta_l_z1) # (784,batch_size) *(batch_size, 256)
        self.grads_b1 = np.sum(delta_l_z1, axis=0) # (256,)

        # print("loss:{} batch_acc:{} lr:{}".format(batch_loss, batch_acc, self.lr))
        # 更新权重
        self.W4 -= self.lr * self.grads_W4
        self.b4 -= self.lr * self.grads_b4
        self.W3 -= self.lr * self.grads_W3
        self.b3 -= self.lr * self.grads_b3
        self.W2 -= self.lr * self.grads_W2
        self.b2 -= self.lr * self.grads_b2
        self.W1 -= self.lr * self.grads_W1
        self.b1 -= self.lr * self.grads_b1
        return batch_loss,batch_acc


if __name__ == '__main__':
    # 训练网络
    net = Network(input_size=784, hidden_size=256, output_size=10, lr=0.2)
    batch_size = 64
    for epoch in range(20):
        losses = []
        accuracies = []
        p_bar = tqdm(range(0, len(X_train), batch_size))
        for i in p_bar:
            x_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            loss,acc = net.step(x_batch,y_batch)
            losses.append(loss)
            accuracies.append(acc)
            # 单batch损失值和准确率
            p_bar.set_description(f"epoch:{epoch+1}, loss:{loss:.4f}, acc:{acc*100:.4f}%")
        # 验证
        y_pred = net.forward(X_val)
        # 单个epoch后验证集平均损失值和准确率
        loss = np.mean(loss_fn(y_val, y_pred))
        acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_val, axis=1))
        print(f"loss:{loss:.4f}, batch_acc:{acc*100:.2f}%, batch_size:{batch_size}, lr:{net.lr}")
    # 测试
    y_pred=net.forward(X_test)
    loss=np.mean(loss_fn(y_test,y_pred))
    acc=np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    # 训练完成后测试集损失值和准确率
    print(f"test ==> loss:{loss:.4f}, batch_acc:{acc*100:.2f}% ,lr:{net.lr}")
```

- 总结
  - 工作目录的检验
  - 前向传播,反向传播的流程
  - 计算梯度,更新参数
  - 求导环节,画出对应示意图



> 这段代码是用于训练一个多层感知器（MLP）模型，以识别MNIST数据集中的手写数字。MNIST数据集包含了手写数字的灰度图像和对应的标签（0-9）。让我们逐行解析这个脚本的功能和组成部分：
>
> ```python
> import torch
> import torch.nn as nn
> import torch.nn.functional as F
> import torch.optim as optim
> from torch.utils.data import Dataset, DataLoader
> import numpy as np
> from matplotlib import pyplot as plt
> import os
> ```
> 这部分代码导入了所需的库。`torch`是PyTorch的主库，用于构建和训练神经网络。`torch.nn`、`torch.nn.functional`和`torch.optim`分别用于定义网络层、提供激活函数和优化器。`Dataset`和`DataLoader`类用于处理和加载数据。`numpy`用于数据处理，`matplotlib.pyplot`用于绘图，`os`用于处理文件路径和目录。
>
> ```python
> print("Current working directory:", os.getcwd())
> x = 'C:/Users/exqin/Desktop/hw2'
> os.chdir(x)
> print("Current working directory:", os.getcwd())
> ```
> 这几行代码打印当前工作目录，然后改变工作目录到指定路径，并再次打印以确认目录已更改。
>
> ```python
> X_train = np.load('./mnist/X_train.npy')
> y_train = np.load('./mnist/y_train.npy')
> X_val = np.load('./mnist/X_val.npy')
> y_val = np.load('./mnist/y_val.npy')
> X_test = np.load('./mnist/X_test.npy')
> y_test = np.load('./mnist/y_test.npy')
> ```
> 加载训练集、验证集和测试集数据及其对应的标签。数据以NumPy数组的形式存储。
>
> ```python
> class MNISTDataset(Dataset):
>     def __init__(self, data=X_train, label=y_train):
>         ...
>     def __getitem__(self, index):
>         ...
>     def __len__(self):
>         ...
> ```
> 定义了一个继承自`Dataset`的类，用于封装MNIST数据，以便`DataLoader`能够加载数据。`__init__`方法初始化数据集，`__getitem__`方法根据索引返回单个样本，`__len__`方法返回数据集的总大小。
>
> ```python
> class Net(nn.Module):
>     def __init__(self):
>         super(Net, self).__init__()
>         ...
>     def forward(self, x):
>         ...
> ```
> 定义了网络结构。该网络包含三个全连接层，使用ReLU激活函数，最后一层不使用激活函数而是直接返回结果，用于计算log_softmax。
>
> ```python
> model = Net()
> model.to(device='cuda')
> ```
> 实例化模型并将其移至GPU以加速训练。
>
> ```python
> criterion = nn.CrossEntropyLoss()
> optimizer = optim.SGD(model.parameters(), lr=0.001)
> ```
> 定义了损失函数和优化器。这里使用交叉熵损失和随机梯度下降优化器。
>
> ```python
> train_loader = DataLoader(MNISTDataset(X_train, y_train), batch_size=64, shuffle=True)
> val_loader = DataLoader(MNISTDataset(X_val, y_val), batch_size=64, shuffle=True)
> test_loader = DataLoader(MNISTDataset(X_test, y_test), batch_size=64, shuffle=True)
> ```
> 创建了训练、验证和测试数据的`DataLoader`，这些`DataLoader`能够批量加载数据，同时提供数据打乱的功能。
>
> 接下来的部分是训练循环，其中包括设置模型为训练模式，遍历训练数据加载器中的数据，执行前向传播、计算损失、执行后向传播和参数更新。
>
> 然后，将模型设置为评估模式，遍历验证数据加载器中的数据，计算模型在验证集上的性能。
>
> 最后，绘制训练和验证过程中的损失和准确率，以便于观察模型的学习情况。
>
> 整个脚本的目的是通过调整网络结构和参数，比如增加隐藏层和改变优化器，以及添加诸如Dropout之类的正则化方法，观察模型在MNIST数据集上的训练效果，尝试达到或超过98%的测试集准确率。



## `Numpy`

### 请为我介绍`Numpy`的广播机制?

> NumPy 的广播机制是一个非常强大的功能，它允许不同形状的数组进行数学运算。广播的核心思想是在适当的条件下，用较小的数组在较大数组的某个维度上进行复制，使它们的形状相匹配，从而可以进行元素级的运算。
>
> ### 广播的规则
>
> NumPy在执行操作时遵循以下规则进行广播：
>
> 1. **维度对齐**：比较两个数组的形状，从末尾维度（最右边）开始比较，每个维度大小需要满足以下条件之一：
>    - 相等。
>    - 其中一个为1。
>    - 其中一个数组的该维度不存在。
>
> 2. **维度扩展**：在较小的数组形状前面补1，直到两个数组的维度数相同。
>
> 3. **大小为1的维度扩展**：在比较的维度中，大小为1的维度会被扩展（复制）以匹配另一个数组的大小。
>
> ### 广播的例子
>
> **例1**：一个数组加上一个数字
>
> ```python
> import numpy as np
> 
> a = np.array([1, 2, 3])
> b = 2
> print(a + b)
> ```
>
> 输出将会是 `[3, 4, 5]`。这里，数字`b`被“广播”到数组`a`的所有元素上。
>
> **例2**：两个数组形状不完全相同
>
> ```python
> import numpy as np
> 
> a = np.array([[0], [10], [20], [30]])
> b = np.array([0, 1, 2])
> 
> print(a + b)
> ```
>
> 输出将会是：
>
> ```
> [[ 0  1  2]
>  [10 11 12]
>  [20 21 22]
>  [30 31 32]]
> ```
>
> 这里，`a`的形状是`(4, 1)`，而`b`的形状是`(3,)`。根据广播规则：
>
> - 首先，`b`的形状在前面补1，变为`(1, 3)`。
> - 然后，两个数组的形状都在不足的维度上扩展，`a`变为`(4, 3)`，`b`也变为`(4, 3)`。
> - 最后，两个数组可以进行元素级的加法操作。
>
> ### 注意事项
>
> - 广播机制使得数组操作更加高效，但也需要注意，它可能导致意外的内存使用增加，因为它会在需要的时候创建较大数组的临时副本。
> - 不是所有的操作都可以通过广播来完成，如果无法按照上述规则匹配数组形状，NumPy将抛出一个异常。
>
> 理解和熟练使用NumPy的广播机制，可以极大地提高数据处理的效率和代码的简洁性。

### 请为我详细介绍`Numpy`中的形状函数shape对应的形状

> 在 NumPy 中，`shape` 是一个非常基础且重要的概念，它表示数组的维度和每个维度的大小。了解一个数组的形状对于进行数据分析和数学运算至关重要。
>
> ### 形状（Shape）
>
> - **形状**：一个数组的形状由一系列的整数定义，这些整数表示数组在每个维度上的大小。形状是一个元组（tuple），其中包含的每个元素代表该维度上的元素数量。
>
> ### 如何获取形状
>
> 你可以通过数组的 `.shape` 属性来访问其形状：
>
> ```python
> import numpy as np
> 
> a = np.array([[1, 2, 3], [4, 5, 6]])
> print(a.shape)  # 输出: (2, 3)
> ```
>
> 在这个例子中，`a` 是一个 2x3 的二维数组，因此它的形状为 `(2, 3)`。
>
> ### 形状的组成
>
> - 对于一维数组（向量），形状会是 `(n,)`，其中 `n` 是数组中元素的数量。
> - 对于二维数组（矩阵），形状是 `(m, n)`，其中 `m` 是行数，`n` 是列数。
> - 对于更高维度的数组，形状会依次包含更多的维度大小。
>
> ### 示例
>
> 1. **一维数组**：
>
>     ```python
>     arr1d = np.array([1, 2, 3, 4, 5])
>     print(arr1d.shape)  # 输出: (5,)
>     ```
>
> 2. **二维数组**：
>
>     ```python
>     arr2d = np.array([[1, 2, 3], [4, 5, 6]])
>     print(arr2d.shape)  # 输出: (2, 3)
>     ```
>
> 3. **三维数组**：
>
>     ```python
>     arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
>     print(arr3d.shape)  # 输出: (2, 2, 2)
>     ```
>
> ### 修改形状
>
> NumPy 也允许你改变数组的形状，而不改变其数据，这可以通过 `reshape` 方法实现：
>
> ```python
> a = np.arange(6)  # 创建一个形状为 (6,) 的数组
> print(a)  # 输出: [0 1 2 3 4 5]
> 
> b = a.reshape((2, 3))  # 将其重新形状为 2x3 的二维数组
> print(b)
> # 输出:
> # [[0 1 2]
> #  [3 4 5]]
> ```
>
> ### 注意事项
>
> - 形状修改操作，如 `reshape`，通常返回数组的视图（view），而不是复制数据。这意味着，原数组和新形状的数组共享相同的数据缓冲区。
> - 如果指定的新形状与原始数组的元素总数不匹配，`reshape` 操作将失败。
>
> 理解并熟悉 NumPy 数组的形状及其操作对于有效地使用 NumPy 进行数据处理和分析至关重要。

### 一般来说(n,)对应的是行向量还是列向量,它与(1,n)有什么区别?

> 在数学和线性代数中，行向量和列向量的区别主要体现在它们的方向和在矩阵中的表示方法上。当我们使用 NumPy 或其他类似的科学计算库时，这种区别也有所体现，尽管在一维数组 `(n,)` 的情况下，并不直接指明是行向量还是列向量。
>
> ### 一维数组 `(n,)`
>
> - 当我们谈论一个形状为 `(n,)` 的 NumPy 数组时，我们指的是一个一维数组，它包含 `n` 个元素。**这种数组既不是行向量也不是列向量；它仅仅是一个一维序列。**
>
>   ==需要通过`reshape`或者广播机制使其维度升高之后才可以进行矩阵级别的操作==
>
> - 在某些上下文中，人们可能会根据操作或预期的应用将其视为行向量或列向量，但从 NumPy 的角度来看，它只是一个一维数组。
>
> ### 二维数组 `(1, n)` 和 `(n, 1)`
>
> - 形状为 `(1, n)` 的数组可以被视为一个**行向量**，因为它表示一个只有一行但有多列的二维数组。
> - 形状为 `(n, 1)` 的数组可以被视为一个**列向量**，因为它表示一个只有一列但有多行的二维数组。
>
> ### 区别
>
> - **一维数组 `(n,)`**：
>   - 它只有一个维度，因此不直接表明行向量或列向量的概念。
>   - 在进行某些数学操作时，尤其是涉及到矩阵运算时，可能需要明确地将其转换为二维的行向量或列向量形式。
>
> - **二维数组 `(1, n)`（行向量）和 `(n, 1)`（列向量）**：
>   - 它们是明确的二维结构，更接近于线性代数中行向量和列向量的定义。
>   - 在矩阵运算中，行向量和列向量的区别非常重要，比如在矩阵乘法中。
>
> ### 转换示例
>
> 如果你有一个形状为 `(n,)` 的数组，并希望明确地使用它作为行向量或列向量，你可以通过 `reshape` 方法来改变其形状：
>
> ```python
> import numpy as np
> 
> # 创建一个形状为 (4,) 的一维数组
> a = np.array([1, 2, 3, 4])
> 
> # 将其转换为形状为 (1, 4) 的行向量
> row_vec = a.reshape(1, -1)
> print("Row vector:", row_vec)
> 
> # 将其转换为形状为 (4, 1) 的列向量
> col_vec = a.reshape(-1, 1)
> print("Column vector:", col_vec)
> ```
>
> 这种转换让你能够明确数组的用途（作为行向量还是列向量），并确保在进行矩阵和向量运算时符合预期。

###  `s = x.reshape(-1, 1)` 这句话会把x变成什么形状?

> 当你在 NumPy 中使用 `x.reshape(-1, 1)` 时，你实际上是在将数组 `x` 重塑为一个二维数组，其中该数组有一个列和足够多的行来容纳原始数组中的所有元素。
>
> - ==`-1` 在 `.reshape()` 方法中作为一个参数表示该维度的大小应该被自动计算，以便保持数组中元素的总数不变。==
> - `1` 表示新数组的第二维（列的数量）应该是 1。
>
> 因此，不论 `x` 的原始形状如何（只要 `x` 是一个一维数组），`x.reshape(-1, 1)` 都会生成一个形状为 `(n, 1)` 的数组，其中 `n` 是 `x` 中元素的总数。这实质上把 `x` 转换为了一个列向量。
>
> ### 示例
>
> ```python
> import numpy as np
> 
> # 假设 x 是一个包含 6 个元素的一维数组
> x = np.array([1, 2, 3, 4, 5, 6])
> 
> # 使用 .reshape(-1, 1) 将 x 转换为一个列向量
> s = x.reshape(-1, 1)
> 
> print(s)
> # 输出将会是：
> # [[1]
> #  [2]
> #  [3]
> #  [4]
> #  [5]
> #  [6]]
> ```
>
> 在这个例子中，`s` 的形状将会是 `(6, 1)`，意味着它有 6 行和 1 列。

### `axis=`参数

> 在 NumPy 中，`axis` 参数用于指定数组操作的轴（axis）。轴可以理解为数组的某个维度，而 `axis` 的序号决定了操作的方向。对于一个给定的数组，不同的 `axis` 值对应于数组的不同维度。
>
> 以下是 `axis` 对应的序号和它们所代表的操作方向：
>
> - `axis=0`：第一个维度，也就是数组的行方向（对于二维数组来说，就是沿着行的方向）。对应每一列
> - `axis=1`：第二个维度，也就是数组的列方向（对于二维数组来说，就是沿着列的方向）。对应每一行
> - `axis=2`：第三个维度，对于三维数组来说，可以理解为“页”的方向。
> - 以此类推，`axis=n` 表示数组的第 `n` 个维度。
>
> 让我们通过一些具体的例子来说明 `axis` 参数的使用：
>
> ### 二维数组（矩阵）
>
> 假设我们有一个二维数组（矩阵）`A`，其形状为 `(m, n)`，即 `A` 有 `m` 行和 `n` 列。
>
> - `np.sum(A, axis=0)`：计算每一列的和，结果是一个形状为 `(n,)` 的一维数组，每个元素代表对应列的和。
> - `np.sum(A, axis=1)`：计算每一行的和，结果是一个形状为 `(m,)` 的一维数组，每个元素代表对应行的和。
>
> ### 三维数组
>
> 假设我们有一个三维数组 `B`，其形状为 `(p, q, r)`，即 `B` 有 `p` 层，每层有 `q` 行和 `r` 列。
>
> - `np.sum(B, axis=0)`：计算每个“页”（`q x r` 子矩阵）的和，结果是一个形状为 `(q, r)` 的二维数组，每个元素代表对应“页”的和。
> - `np.sum(B, axis=1)`：计算每层中每一行的和，结果是一个形状为 `(p, r)` 的二维数组，每个元素代表对应层中对应行的和。
> - `np.sum(B, axis=2)`：计算每层中每一列的和，结果是一个形状为 `(p, q)` 的二维数组，每个元素代表对应层中对应列的和。
>
> ### 更高维度的数组
>
> 对于更高维度的数组，`axis` 的概念是类似的。`axis` 的值告诉 NumPy 沿着哪个维度进行操作。
>
> ### 注意事项
>
> - 当使用 `axis` 参数时，要确保你的操作是沿着正确的维度进行的，否则可能会得到意想不到的结果。
> - 在某些函数中，如 `np.mean`、`np.std`、`np.dot` 等，`axis` 参数也用于指定沿着哪个维度进行计算或减少维度。
> - 在使用 `axis` 参数时，要记住 NumPy 是按 C 语言风格进行索引的，即第一个维度是 `axis=0`。
>
> 通过正确使用 `axis` 参数，你可以在 NumPy 中高效地进行各种数组操作。
>
> ![e0e7c0e870791c6da20a4c5a5ced14a](./aiNote.assets/e0e7c0e870791c6da20a4c5a5ced14a.jpg)

## `Pytorch`

## `matplotlib`



## 常见激活函数的实现

### `Sigmod`函数

>  Sigmoid 函数是一个经典的激活函数，通常用于输出层进行二分类问题的概率预测。

- 函数
  -  $\sigma(x) = \frac{1}{1+e^{-x}}$
- 导数
  - $\sigma^{(1)}(x) = \sigma(x)\times{(1-\sigma(x))} = \frac{e^x}{(e^x+1)^2}$​

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

###  `Relu`函数

> ReLU（Rectified Linear Unit）函数是最常用的隐藏层激活函数之一，因其简单和效果良好而广泛应用。

- 函数
  - $f(x) = max(0,x)$
- 导数
  - $f^{(1)}(x) =\begin{cases} 1&\text{x>0}\\0&\text{otherwise}\end{cases} $



```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype) # 前面是一个布尔表达式,后面的astype方法可以使得函数返回一个和x数据类型的数据,通过广播机制得到相应矩阵
```

### `Softmax`函数

> `Softmax` 函数常用于多分类问题的输出层，它将多个线性输出转换为概率分布。

- 函数
  - 对于向量 (x) 的第 (i) 个元素,$Softamx(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{N}e^{x_j}}$

-  导数
  - 一般**联合交叉熵损失函数**求导

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 减去max(x)增加数值稳定性
    return exp_x / np.sum(exp_x, axis=0)

# Softmax 函数的导数较为复杂，依赖于具体的输出和目标值，这里不做展开
```

### `Tanh`函数

> Tanh（双曲正切）函数是 Sigmoid 函数的变种，它将输出值范围调整为了(-1,1).

- 函数
  - $tanh(x) = \cfrac{e^x-e^{-x}}{e^x+e^{-x}}$ 
- 导数
  - $tanh^{(1)}(x)=1-tanh^2(x)$

```python 
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

## 常见优化方法

### SGD（随机梯度下降）

SGD 是最基本的优化算法之一，它按照以下步骤更新每个参数：

1. 计算损失函数关于每个参数的梯度。
2. 对每个参数减去学习率乘以其梯度（`parameter -= learning_rate * gradient`）。

SGD 的一个关键特点是它的**学习率是固定**的，这意味着在整个训练过程中，参数更新的步长保持不变。这可能会导致一些问题，比如收敛速度慢，或者在最优解附近震荡。

### Adam（Adaptive Moment Estimation）

Adam 是一种自适应学习率的优化算法，它结合了动量（Momentum）和 RMSprop 的优点。Adam 通过计算每个参数的自适应学习率来更新参数，这使得它在实践中通常比 SGD 更快地收敛。

Adam 的参数更新规则如下：

1. 计算梯度。
2. 计算梯度的一阶矩估计（动量）和二阶矩估计（未中心化的方差）。
3. 对一阶矩估计和二阶矩估计进行偏差校正。
4. 使用这些估计值来计算每个参数的自适应学习率。
5. 使用自适应学习率更新参数。

**Adam 的关键优势在于它自适应地调整每个参数的学习率，这通常有助于更快地收敛，并且对于不同的参数，它可以有不同的学习速率。**



### RMSprop

 RMSprop（Root Mean Square Propagation）是一种自适应学习率的优化算法，由 Geoffrey Hinton 在他的课程中提出。它是对标准随机梯度下降（SGD）的一种改进，旨在解决学习率选择困难和梯度消失或爆炸的问题。**RMSprop 特别适用于训练具有非平稳目标函数和稀疏梯度的神经网络。**

RMSprop 的核心思想是为每个参数维护一个独立的学习率，这些学习率会根据参数的历史梯度平方的均值自动调整。这样做可以确保每个参数的更新步长保持稳定，即使在梯度变化很大的情况下。

RMSprop 的参数更新规则如下：

1. 对于每个参数，计算其梯度。
2. 对于每个参数，维护一个时间窗口内的梯度平方的移动平均（称为缓存）。
3. 更新缓存：`cache = decay_rate * cache + (1 - decay_rate) * gradient_squared`，其中 `gradient_squared` 是当前梯度的平方。
4. 计算每个参数的学习率：`learning_rate = learning_rate / (sqrt(cache) + epsilon)`，其中 `learning_rate` 是初始学习率，`epsilon` 是一个很小的常数，以防止除以零。
5. 使用这个学习率更新参数：`parameter = parameter - learning_rate * gradient`。

RMSprop 的关键优势在于它的自适应性，它通过调整每个参数的学习率来适应梯度的变化。`decay_rate` 参数用于控制缓存的衰减，以防止学习率过快地适应到梯度的短期波动。

RMSprop 与其他自适应学习率优化算法（如 Adam）相比，它的实现更简单，计算效率更高。然而，它也有一些局限性，例如，它可能不适用于所有类型的数据集和网络结构。在实践中，RMSprop 通常与其他优化算法一起使用，以便找到最适合特定任务的优化策略。

## 常见损失函数的实现

## 过拟合的应对措施

### 惩罚函数



### Dropout

Dropout 是一种在神经网络训练过程中使用的正则化技术，由 Geoffrey Hinton 和他的同事在 2012 年提出。**Dropout 的主要目的是防止神经网络的过拟合，即模型在训练数据上表现很好，但在未见过的数据上表现不佳的现象。**

Dropout 的原理非常简单：在每次训练迭代中，随机地“丢弃”（即暂时移除）网络中的一些神经元（以及它们的连接），这样就可以防止网络对特定的神经元过度依赖。通过这种方式，Dropout 强制网络学习更加鲁棒的特征，因为每个神经元必须能够在没有其他神经元的情况下也能正常工作。

具体来说，Dropout 按照以下步骤进行：

1. 在每次训练迭代开始时，为每个神经元生成一个 [0, 1] 范围内的随机数。
2. 如果这个随机数小于预设的 Dropout 概率（通常是一个小于 1 的值，如 0.5），那么这个神经元在这次迭代中被“丢弃”，即它的输出被设置为 0，并且不会参与前向传播和反向传播。
3. 如果随机数大于或等于 Dropout 概率，那么神经元正常工作，参与前向传播和反向传播。
4. 在每次迭代结束后，更新模型的权重（如果使用 SGD 或其他优化算法）。
5. 在测试或评估模型时，不使用 Dropout，即所有的神经元都参与前向传播和反向传播。为了补偿训练时的 Dropout 效应，通常会将每个神经元的输出乘以 Dropout 概率的倒数，以确保期望输出不变。

Dropout 的效果类似于集成学习中的 Bagging 方法，其中多个模型（或神经元）的平均输出被用作最终的预测。通过这种方式，Dropout 可以减少模型的复杂度，提高泛化能力，并在许多不同的任务和数据集上取得了显著的性能提升。

## 一些坑

### 工作目录的检验

> 如果工作目录不正确会导致`numpy`的`load`函数找不到相应的文件

```python
# 导入os模块
import os
# 输出当前工作目录
print("Current working directory:", os.getcwd())
# 变更工作目录到x
x = 'C:/Users/exqin/Desktop/hw2'
# 用于解决工作路径不正确导致的file not found
os.chdir(x)
```

### 输出的重定向

```python
import sys

# 打开一个文件用于写入，模式为 'w' (写入模式，如果文件存在则覆盖)
with open('output.txt', 'w') as file:
    # 重定向标准输出到这个文件
    sys.stdout = file
    print("这条信息会被写入到文件中，而不是显示在控制台。")
```

```python
import sys
# 打开一个文件用于写入，模式为 'w' (写入模式，如果文件存在则覆盖)
file = open('output.txt', 'w')
# 重定向标准输出到这个文件
sys.stdout = file

print("这条信息会被写入到文件中，而不是显示在控制台。")
```

> 在 Python 中，如果你想要同时将输出显示在控制台并且记录到文件中，你可以使用 `tee` 函数。`tee` 函数是 `sys` 模块的一部分，它允许你将输出流分发到多个地方。
>
> 以下是一个示例，展示了如何将标准输出同时重定向到控制台和文件：
>
> ```python
> import sys
> from io import StringIO
> 
> # 创建一个 StringIO 对象，它可以被用作文件对象
> string_io = StringIO()
> 
> # 将标准输出重定向到 StringIO 对象和实际的控制台输出
> sys.stdout = sys.stdout.__class__(sys.stdout.write, sys.stdout.flush, string_io.write)
> 
> # 定义一个辅助函数，用于捕获输出并将其写入文件
> def capture_output():
>     string_io.flush()  # 确保 StringIO 对象中的输出被刷新
>     with open('output.txt', 'a') as file:
>         file.write(string_io.getvalue())  # 将输出写入文件
>     string_io.truncate(0)  # 清空 StringIO 对象的内容
>     string_io.seek(0)  # 重置 StringIO 对象的读写位置
> 
> # 现在你可以正常打印输出，它将同时显示在控制台和写入到文件中
> print("这条信息会显示在控制台和追加到 output.txt 文件中。")
> 
> # 捕获并保存输出到文件
> capture_output()
> 
> # 如果你想要在每次打印后都立即将输出写入文件，你可以重写 flush 方法
> class TeeStream:
>     def __init__(self, stream):
>         self.stream = stream
> 
>     def write(self, data):
>         self.stream.write(data)
>         sys.stdout.write(data)  # 保持原有的控制台输出
> 
>     def flush(self):
>         self.stream.flush()
>         sys.stdout.flush()  # 保持原有的控制台输出
> 
> # 使用 TeeStream 重定向输出
> sys.stdout = TeeStream(string_io)
> 
> # 现在你可以正常打印输出，它将同时显示在控制台和写入到文件中
> print("再次打印，输出同样会显示在控制台和追加到 output.txt 文件中。")
> ```
>
> 在这个例子中，我们首先创建了一个 `StringIO` 对象来捕获输出。然后我们定义了一个 `TeeStream` 类，它继承自 `sys.stdout` 的类，并重写了 `write` 和 `flush` 方法。这样，当我们调用 `print` 函数时，输出会被同时发送到 `StringIO` 对象和控制台。
>
> 我们定义了一个 `capture_output` 函数来将 `StringIO` 对象中的内容追加到文件中，并清空 `StringIO` 对象以便下一次捕获输出。在每次打印后，你可以调用 `capture_output` 函数来保存输出到文件。
>
> 请注意，这种方法会在每次打印后立即将输出写入文件，而不是在程序结束时。如果你希望在程序结束时一次性写入文件，你可以在程序的最后调用 `capture_output` 函数。

## 卷积和池化

卷积（Convolution）和池化（Pooling）是卷积神经网络（CNN）中的两个基本操作，它们在图像处理和特征提取中起着至关重要的作用。

1. **卷积的目的**：
   - **特征提取**：卷积层通过使用一组可学习的滤波器（或卷积核）来提取输入图像的局部特征。这些滤波器可以捕捉到边缘、角点、纹理等低级和高级特征。
   - **参数共享**：在卷积操作中，同一个滤波器在整个输入图像上移动并应用，这种参数共享机制大大减少了模型的参数数量，提高了计算效率。
   - **空间不变性**：由于卷积层的滑动窗口性质，模型能够学习到具有空间不变性的特征，即不管特征在图像中的位置如何，模型都能够识别出来。

2. **池化的目的**：
   - **降低维度**：池化操作通过减少数据的空间尺寸来降低模型的复杂度和计算量。这通常通过在特征图上滑动一个窗口并提取最大值（最大池化）或平均值（平均池化）来实现。
   - **抗过拟合**：池化有助于提供一种形式的正则化，因为它在一定程度上增加了输入数据的不变性。这有助于模型对小的位置变化和噪声更加鲁棒，从而减少过拟合的风险。
   - **捕获主要特征**：池化操作还可以帮助模型关注更重要的特征，忽略不重要的细节，从而在保留关键信息的同时减少噪声。

总的来说，卷积和池化在卷积神经网络中共同工作，以有效地从图像数据中提取有用的特征，并为后续的分类或回归任务提供强大的表示。卷积负责提取局部特征，而池化则负责降维和提高特征的鲁棒性。



卷积层和池化层（也称为下采样层）的输出尺寸可以通过以下公式计算得出：

**卷积层输出尺寸公式：**

$ \text{Output Width} = \left\lfloor \frac{\text{Input Width} + 2 \times \text{Padding} - \text{Kernel Width}}{\text{Stride}} + 1 \right\rfloor $



$ \text{Output Height} = \left\lfloor \frac{\text{Input Height} + 2 \times \text{Padding} - \text{Kernel Height}}{\text{Stride}} + 1 \right\rfloor $​



其中：

- 输入宽度（Input Width）和输入高度（Input Height）是输入图像的宽度和高度。
- 卷积核宽度（Kernel Width）和卷积核高度（Kernel Height）是卷积核的大小。
- 步长（Stride）是卷积时的步长。
- 填充（Padding）是卷积层中在输入图像周围添加的零填充的数量。

**池化层输出尺寸公式：**

$text{Output Width} = \left\lfloor \frac{\text{Input Width}}{\text{Stride}} \right\rfloor $​



$text{Output Height} = \left\lfloor \frac{\text{Input Height}}{\text{Stride}} \right\rfloor $​



其中：

- 输入宽度（Input Width）和输入高度（Input Height）是输入图像的宽度和高度。
- 步长（Stride）是池化时的步长。

在这些公式中，`leftfloor{}` 表示向下取整，即只保留整数部分。这是因为卷积和池化操作可能会使输出尺寸小于输入尺寸。

请注意，如果卷积层的填充和步长选择得当，可以使得输出尺寸保持不变（例如，当卷积核大小为 3，步长为 1，填充为 1 时）。同样，如果池化层的步长为 2，那么输出尺寸将是输入尺寸的一半。在实际应用中，通常需要根据网络的设计和所需的输出尺寸来调整这些参数。



## 增强泛化性(减小过拟合)

在机器学习中，正则化是一种避免模型过拟合的技术，它通过在损失函数中添加一个额外的项来惩罚模型复杂度。以下是一些常见的正则化方法：

1. **L1正则化（Lasso）**：
   - L1正则化通过向损失函数添加权重参数的绝对值之和来工作。这种方法可以产生稀疏权重矩阵（即许多权重为0），从而进行特征选择。
   - 公式表示：\( $L1(\mathbf{w}) = \lambda \sum_{i=1}^{n} |w_i| $，其中 \( $\mathbf{w}$ \) 是权重向量，\( $\lambda$ \) 是正则化强度。

2. **L2正则化（Ridge）**：
   - L2正则化通过添加权重参数的平方和来工作，这有助于减少权重的值，但不会使它们变为0，因此与L1正则化相比，它不具备特征选择的能力。
   - 公式表示：$ L2(\mathbf{w}) = \lambda \sum_{i=1}^{n} w_i^2 $。

3. **弹性网（Elastic Net）**：
   - 弹性网是L1和L2正则化的组合，它结合了两者的优点，同时进行特征选择和权重缩减。
   - 公式表示：$ L1 + \alpha L2(\mathbf{w}) = \lambda \sum_{i=1}^{n} |w_i| + \alpha \sum_{i=1}^{n} w_i^2 $，其中 \( $\alpha$ \) 控制L1和L2项的相对贡献。

4. **Dropout**：
   - Dropout在神经网络训练过程中随机地丢弃（即暂时移除）一些神经元的输出，这有助于防止模型对特定的神经元过度依赖，从而提高模型的泛化能力。

5. **早停（Early Stopping）**：
   - 早停是指在训练过程中，一旦验证集上的性能不再提升或开始下降，就停止训练。这有助于避免模型在训练集上过度拟合。

6. **批量归一化（Batch Normalization）**：
   - 批量归一化通过对每一层的激活值进行规范化来减少内部协变量偏移，这有助于提高训练的稳定性和效率。

7. **数据增强（Data Augmentation）**：
   - 数据增强通过对训练数据应用各种变换（如旋转、缩放、裁剪等）来人为地增加数据集的大小和多样性，从而提高模型的泛化能力。

这些正则化方法可以单独使用，也可以组合使用，以适应不同的模型和数据集。选择哪种正则化方法取决于具体问题、数据的特性以及模型的复杂度。

### 归一化

> 归一化在深度学习中扮演着非常重要的角色，它有助于提高训练过程的稳定性、效率和模型的最终性能。归一化主要有两种类型：数据归一化和批量归一化。
>
> ### 数据归一化（Data Normalization）
>
> 数据归一化通常是指对输入数据进行预处理，使其具有统一的尺度。这有助于加速训练过程，并确保模型中的不同参数具有相似的初始尺度。数据归一化的常见方法包括：
>
> - **最小-最大归一化**（Min-Max Normalization）：将数据缩放到一个固定的范围内，通常是 [0, 1] 或 [-1, 1]。
> - **Z得分归一化**（Z-Score Normalization）：将数据的均值变为 0，标准差变为 1。
> - **小数定标归一化**（Decimal Scaling）：通过移动数据的小数点位置来进行归一化。
>
> 数据归一化有助于改善梯度下降算法的收敛速度和稳定性，因为它确保了所有特征都在相同的尺度上，避免了某些特征在计算梯度时主导其他特征。
>
> ### 批量归一化（Batch Normalization）
>
> 批量归一化是一种在网络训练过程中动态进行的归一化技术，它对每个小批量数据进行归一化，并引入了可学习的缩放（gamma）和偏移（beta）参数。批量归一化的主要作用包括：
>
> 1. **减少内部协变量偏移**：在训练过程中，由于参数更新，每一层的输入分布都会发生变化。批量归一化通过规范化激活值来减少这种分布变化，从而使得每一层的输入更加稳定。
>
> 2. **加速训练**：由于梯度不会受到饱和或爆炸的影响，批量归一化可以使得更高的学习率得以应用，从而加速模型的收敛。
>
> 3. **减少过拟合**：批量归一化可以被视为一种正则化技术，因为它为每个小批量数据引入了噪声。这种噪声有助于防止模型对训练数据过度拟合。
>
> 4. **消除参数初始化的依赖**：由于归一化的存在，模型对参数初始化的选择不那么敏感，这使得模型训练更加灵活。
>
> 5. **允许更深的网络**：批量归一化使得训练深层网络成为可能，因为即使在深层网络中，梯度也不会消失或爆炸。
>
> 总的来说，归一化是深度学习中一个关键的预处理和正则化步骤，它有助于提高模型的训练效率和泛化能力。

批量归一化（Batch Normalization，简称 BN）是一种用于提高神经网络训练稳定性和性能的技术。它通过规范化网络中间层的激活值来减少内部协变量偏移（Internal Covariate Shift），这是指网络各层输入分布的变化。

在 PyTorch 中，`nn.BatchNorm2d` 是对 2D 特征图进行批量归一化的工具，通常用于卷积神经网络中。`nn.BatchNorm2d` 的参数 `num_features` 指定了特征图的通道数。

以下是 `nn.BatchNorm2d` 的工作原理：

1. **归一化**：对于每个通道，批量归一化会计算当前批次中所有激活值的均值和标准差，并使用这些统计量对激活值进行规范化，使其均值为 0，标准差为 1。

2. **缩放和偏移**：尽管规范化有助于稳定训练过程，但它也可能导致网络中的信息损失。为了解决这个问题，批量归一化引入了两个可学习的参数，分别用于缩放（scale）和偏移（shift）。这两个参数允许网络恢复那些对最终任务有用的特征。

3. **批量统计量**：批量归一化计算每个批次的均值和方差，而不是使用整个训练集的统计量。这有助于适应数据分布的变化，并减少训练过程中的梯度问题。

在实际使用中，批量归一化层通常位于卷积层和非线性激活函数之间。例如，如果你的网络结构是这样的：

```python
def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(64)  # 与卷积层的输出通道数相匹配
    self.relu = nn.ReLU()
    # ... 其他层的定义
```

在 `forward` 方法中，你会这样使用它们：

```python
def forward(self, x):
    x = self.conv1(x)  # 卷积操作
    x = self.bn1(x)     # 批量归一化
    x = self.relu(x)    # 非线性激活函数
    # ... 其他层的操作
    return x
```

批量归一化有多个变体，如 `nn.BatchNorm1d`（用于全连接层）和 `nn.SyncBatchNorm`（用于同步批量归一化，通常用于数据并行训练）。在大多数情况下，`nn.BatchNorm2d` 是卷积层后的标准选择。

## 优化器

> PyTorch 提供了多种优化器，用于在训练神经网络时更新模型的参数。以下是 PyTorch 中一些常见的优化器：
>
> 1. **SGD（随机梯度下降）**:
>    - `torch.optim.SGD`：标准随机梯度下降优化器。
>    - `torch.optim.SGD` 还支持动量（Momentum），可以通过 `momentum` 参数来启用。
>
> 2. **Adam（自适应矩估计）**:
>    - `torch.optim.Adam`：结合了动量和 RMSprop 的优点，自适应地调整每个参数的学习率。
>
> 3. **AdamW（Adam with weight decay）**:
>    - `torch.optim.AdamW`：Adam 优化器的变体，将 L2 权重衰减纳入考虑。
>
> 4. **RMSprop（均方根传播）**:
>    - `torch.optim.RMSprop`：处理非平稳目标函数的优化器，适用于处理稀疏梯度。
>
> 5. **Adagrad（自适应梯度）**:
>    - `torch.optim.Adagrad`：自适应学习率的优化器，适用于处理稀疏梯度。
>
> 6. **Adadelta（Adagrad的扩展）**:
>    - `torch.optim.Adadelta`：Adagrad 的改进版本，旨在减少其学习率单调递减的问题。
>
> 7. **Adamax（Adam的扩展）**:
>    - `torch.optim.Adamax`：Adam 的变体，使用无穷范数来缩放梯度的每个元素。
>
> 8. **SparseAdam（稀疏Adam）**:
>    - `torch.optim.SparseAdam`：适用于稀疏梯度的 Adam 优化器。
>
> 9. **LBFGS（拟牛顿法）**:
>    - `torch.optim.LBFGS`：有限内存 Broyden–Fletcher–Goldfarb–Shanno 算法，适用于小到中等规模的问题。
>
> 这些优化器都有各自的特点和适用场景。选择哪种优化器取决于具体的问题、数据集和模型架构。在实践中，可能需要尝试多种优化器来找到最适合当前任务的优化策略。
