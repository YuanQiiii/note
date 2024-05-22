## Q&A

### 虚拟环境是什么?

在Python中，虚拟环境是一个独立的目录树，可以在其中安装Python库(别人写好的代码文件,可以用于调用)和应用程序，并且这些安装不会影响到系统中其他Python项目的环境。每个虚拟环境都有自己的Python二进制文件（即可能有不同版本的Python）和一套独立的库。

虚拟环境的使用是Python开发的最佳实践之一，因为它们帮助解决了多个项目之间的依赖冲突问题。如果没有虚拟环境，不同项目所需的库版本可能会相互冲突，导致**依赖地狱（dependency hell）**。

### 包是什么?

在Python中，包（Package）是一种包含Python模块的文件夹。它允许你组织相关的模块到一个目录下，这样可以保持代码的整洁和管理上的便利。包本质上是一个带有特殊文件 `__init__.py` 的目录。这个文件可以为空，它的存在告诉Python这个目录应该被视为一个Python包，这样你就可以按照目录结构来组织模块，并使用点（.）符号来进行导入。

例如，假设你有这样的目录结构：

```
mypackage/
    __init__.py
    submodule1.py
    submodule2.py
    subpackage/
        __init__.py
        submodule3.py
```

在这个结构中，`mypackage` 是一个包，包含了两个子模块：`submodule1` 和 `submodule2`。包内还有一个名为 `subpackage` 的子包，它自己也有一个 `__init__.py` 文件和一个 `submodule3` 模块。

要导入 `mypackage` 包中的 `submodule1`，你可以这样做：

```python
import mypackage.submodule1
```

或者如果你想只导入 `submodule1` 中的特定函数 `my_function`，你可以这样做：

```python
from mypackage.submodule1 import my_function
```

使用包的好处是：

1. **命名空间**：包提供了避免模块名称冲突的方法。不同的包可以有相同名称的模块，因为它们的完整路径不同。
2. **可维护性**：包允许你以逻辑和有组织的方式来组织代码。
3. **可重用性**：打包为标准格式可以在不同的项目之间重复使用代码。
4. **共享和分发**：包是分享和分发Python代码的标准方式，通常通过Python包索引（PyPI）等平台进行分发。

Python还有一个内置的包管理系统 `pip`，可以让你轻松地安装、升级、和移除第三方包。这些第三方包可以大大扩展Python的功能，它们通常都是按照上面描述的包结构组织的。

## `Anaconda` 

### 环境准备

- 安装`Anaconda`并配置**环境变量**
- 安装`vscode`

### 虚拟环境的使用

- 创建

可以使用如下命令

```cmd
REM NameOfYourEnv是你对于虚拟环境的命名
conda create -n NameOfYourEnv python=x.x
REM 以下是一个例子
conda create -n aibasis python=3.9
```

- 激活

```cmd
REM NameOfYourEnv是你虚拟环境的名称
conda activate NameOfYourEnv
REM 以下是一个例子
conda activate aibasis
```

激活之后，会在路径前面有一个小括号，括号内就是当前正在使用的虚拟环境。

![image-20240225100155445](./aibasisNote.assets/image-20240225100155445.png)

如果我们想要在不同的环境之间跳转，则直接用上面的方式激活另一个环境即可。
如果想要退出虚拟环境，则输入

```cmd
conda deactivate
```

在命令行的右下角，有一个`base:conda`,这是最基础的虚拟环境,是`Anaconda`自带的,一般不在这个环境里安装包

### 包的使用

```cmd
REM 有两种安装方式,优先使用第一种

conda install xxx
pip install xxx
```

> - 更换`conda`和`pip`的镜像源(解决安装慢的问题)
>
> 善用搜索引擎即可解决
>
> - 开发环境：`conda` VS `pip`
>   - **`conda`：**
>     -  环境管理器，高于Python 
>     - 可安装python包、其他库
>     - 包的版本和依赖严谨，会考虑底层库依赖
>     - Python包的数量是`pypi`的子集
>     -  `Pytorch`、`PaddlePaddle`、`numpy`、`scipy`、`qt`等库优先使用`conda`安装
>   - **`pip`：**
>     - Python的一个库
>     - 只能安装python库
>     - 包只有版本号，不会考虑底层库依赖
>     - 所有Python包
>     - `TensorFlow`、`opencvpython`推荐使用`pip`安装

## `NumPy`&`Pytorch`

- NumPy是Python中用于科学计算的核心库之一，旨在提供一个
  强大的数组处理工具，以便更轻松地进行数值计算和数据分析。
- 提供了**高性能**的多维数组对象和用于处理这些数组的函数，使
  得处理大型数据集变得更加简单和高效。(fixed type)
  - SIMD Vector Processing (single instruction multiple data)
  
- NumPy是Python中进行科学计算和数据分析的基础库之一，几
  乎所有与数据相关的Python库都依赖于它。
- [NumPy | 菜鸟教程 (runoob.com)](https://www.runoob.com/numpy/numpy-ndarray-object.html)
- PyTorch 是一个开源的深度学习框架，由 Facebook AI Research（FAIR）开发并维护。它提供了灵活的张量计算和构建深度神经网络的工具，同时具有动态计算图和自动微分的功能。以下是一些 PyTorch 的特点和优势：

  1. **动态计算图**：
     - PyTorch 使用动态计算图，这意味着计算图是按需构建的，可以在运行时进行更灵活的计算图操作。
     - 这与 TensorFlow 的静态计算图相比，更容易调试、理解和编写代码。

  2. **自动微分**：
     - PyTorch 的 autograd 模块提供了自动微分的功能，可以自动计算张量的梯度，简化了反向传播过程。
     - 开发者无需手动计算梯度，只需调用 `.backward()` 方法即可实现自动求导。

  3. **易于学习和使用**：
     - PyTorch 的 API 设计简洁清晰，易于学习和使用，尤其适合深度学习初学者入门。
     - 用户友好的接口和文档使得开发者能够快速上手并快速实现其想法。

  4. **动态神经网络**：
     - PyTorch 支持动态构建和修改神经网络模型，使得实验和研究更加灵活和方便。
     - 开发者可以根据需求在模型中添加、删除或修改层，而无需重新定义整个模型。

  5. **丰富的扩展库**：
     - PyTorch 提供了丰富的扩展库，如 torchvision（用于计算机视觉任务）、torchtext（用于自然语言处理）等。
     - 这些扩展库使得开发者能够更轻松地处理各种深度学习任务。

  6. **支持GPU加速**：
     - PyTorch 提供了对 GPU 的支持，可以利用 GPU 进行张量计算，加速模型训练和推理过程。
     - 通过简单的代码更改，可以将张量移动到 GPU 上进行计算。


### 具体操作

```python
import numpy as np
# 生成数组
a = np.array([1,2,3])
print(a)
# 生成多维数组,注意括号的嵌套
b = np.array([[9.0,8.0,7.0],[6.0,5.0,4.0]])
print(b)
# 获取数组的维度
a.ndim 
print(a.ndim,type(a.ndim))
# 获取数组的形状
b.shape
print(b.shape,type(b.shape))
# 获取数组内部的数据类型,也可以用来指定数组内部元素的数据类型,否则将使用默认值
a.dtype
print(b.dtype,type(b.dtype))
# 获取元素的大小(以字节表示)
a.itemsize
print(a.itemsize,type(a.itemsize))
# 获取数组的总大小
a.nbytes
print(a.nbytes,type(a.nbytes))

a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)
# 对于特定位置的元素[row,column] 输出/赋值
print(a[0,:])
print(a[1,:])
print(a[:,0])
# [start:end:step],可以使用切片
print(a[0,1:6:2])
a[:,2] = [1,2]
print(a)
```

```python
x = 3
# 格式化字符串
print("x = %d" %x)
print("x = {}" .format(7))
print("hello world")
print("x = ", x)

type(x)

str_x = str(x)
type(str_x)

def count():
  y = 1+1
  return 10

print(count())

import numpy as np

a = np.array([1,2])
print(a.shape)
print(a.dtype)

import copy
# 浅拷贝(创建引用)
original_list = [1, [2, 3]]
copied_list = copy.copy(original_list)
original_list[0] = 0
original_list[1][0] = 0
print(copied_list)  # [1, [0, 3]]

# 深拷贝(创建新的变量,在原本上的修改不会反馈到副本上)
original_list = [1, [2, 3]]
deepcopied_list = copy.deepcopy(original_list)
original_list[0] = 0
original_list[1][0] = 0
print(deepcopied_list)  #  [1, [2, 3]]


original_list = [1, [2, 3]]
deepcopied_list = original_list
original_list[0] = 0
original_list[1][0] = 0
print(deepcopied_list)  #  [0, [0, 3]]
```
```python
import numpy as np

# 创建两个矩阵
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# 使用numpy.dot()进行矩阵乘法
result_dot = np.dot(matrix1, matrix2)
print("Result using np.dot():")
print(result_dot)

# 使用@运算符进行矩阵乘法
result_at = matrix1 @ matrix2
print("\nResult using @ operator:")
print(result_at)

# 使用*运算符进行矩阵乘法
result_at = matrix1 * matrix2
print("\nResult using * operator:")
print(result_at)

# 创建一个列表
my_list = [1, 2, 3, 4, 5]

# 创建一个元组
my_tuple = (1, 2, 3, 4, 5)

# 修改列表中的元素
my_list[0] = 0
print(my_list)  # 输出: [0, 2, 3, 4, 5]

# 尝试修改元组中的元素，会引发异常
my_tuple[0] = 0  # TypeError: 'tuple' object does not support item assignment

#new_list = [expression for item in iterable if condition]
squares = [x**2 for x in range(0, 9)]
print(squares)

my_list = []
for x in range(0,9):
  my_list.append(x**2)

print(my_list)

b = [1,2,3,4,5]
print(b[0:2])

# 打开文件
file = open('filename.txt', 'w')

# 写入内容
file.write('Hello, World!')

# 关闭文件
file.close()

with open('filename.txt', 'r') as file:
    content = file.read()
    print(content)

from google.colab import files
import os
print( os.getcwd() )
print( os.listdir() )

files.download('filename.txt')

import pickle

data = {'name': 'Alice', 'age': 30, 'city': 'New York'}

# 序列化对象并写入文件
with open('data.pkl', 'wb') as file:
    pickle.dump(data, file)

import pickle

# 从文件中读取并反序列化对象
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

print(loaded_data)

import json

data = {'name': 'Alice', 'age': 30, 'city': 'New York'}

# 序列化对象并写入文件
with open('data.json', 'w') as file:
    json.dump(data, file)

import json

# 从文件中读取并反序列化对象
with open('data.json', 'r') as file:
    loaded_data = json.load(file)

print(loaded_data)
```
```python
import torch
t = torch.randn(2,2)
print(t.shape)

import numpy as np
a = np.array([2])
print(type(a))
t = torch.from_numpy(a)
print(type(t))

import numpy as np
a = np.array([2])
t = torch.tensor(a)
print(type(t))

t1 = torch.full(t.shape,3)
t1

print(t1.shape)
print(t1.dtype)
print(t1.device)
#t1.gpu()
#t1.cuda()

import torch

# 创建一个3x3的矩阵，元素分别为1到9，然后将其重塑为3x3的形状
matrix = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(3, 3)

# 打印整个矩阵
print("Matrix:")
print(matrix)

# 打印第一行
print("\nFirst row:")
print(matrix[0, 0:1])

# 打印第一列
print("\nFirst column:")
print(matrix[:, 0])

# 打印最后一列
print("\nLast column:")
print(matrix[..., -1])

t = torch.cat([matrix,matrix,matrix],dim = 0)
print(t)

import torch

# 创建一系列张量
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
tensor3 = torch.tensor([7, 8, 9])

# 使用 torch.stack() 将这些张量沿着新的维度进行堆叠
stacked_tensor = torch.stack([tensor1, tensor2, tensor3],dim = 1)

print("Stacked tensor:")
print(stacked_tensor)
print("Shape of stacked tensor:", stacked_tensor.shape)

import torch

# 创建一系列张量
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
tensor3 = torch.tensor([7, 8, 9])

# 使用 torch.stack() 将这些张量沿着新的维度进行堆叠
stacked_tensor = torch.stack([tensor1, tensor2, tensor3],dim = 1)

print("Stacked tensor:")
print(stacked_tensor)
print("Shape of stacked tensor:", stacked_tensor.shape)

import torch

# 定义两个矩阵
matrix1 = torch.tensor([[1, 2], [3, 4]])
matrix2 = torch.tensor([[5, 6], [7, 8]])

# 使用 torch.mm() 函数进行矩阵乘法
result_mm = torch.mm(matrix1, matrix2)

# 使用 torch.matmul() 函数进行矩阵乘法
result_matmul = torch.matmul(matrix1, matrix2)

print("Result using torch.mm():")
print(result_mm)

print("\nResult using torch.matmul():")
print(result_matmul)

import torch

# 定义两个矩阵
matrix1 = torch.tensor([[1, 2], [3, 4]])
matrix2 = torch.tensor([[5, 6], [7, 8]])

# 使用 @ 运算符进行矩阵乘法
result_at = matrix1 @ matrix2

print("Result using @ operator:")
print(result_at)

import torch

# 定义两个张量
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# 使用 torch.mul() 函数进行逐元素相乘
result_mul = torch.mul(tensor1, tensor2)

print("Result using torch.mul():")
print(result_mul)

import torch

# 定义两个张量
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# 使用 * 运算符进行逐元素相乘
result_multiply = tensor1 * tensor2

print("Result using * operator:")
print(result_multiply)

import torch

# 创建一个标量张量
tensor = torch.tensor(3.14)

# 使用 tensor.item() 将标量张量转换为数值
value = tensor.item()

print("Tensor:")
print(tensor)
print("Value:")
print(value)
print("Type of value:", type(value))

import torch

# 创建一个包含多个元素的张量
tensor = torch.tensor([1, 2, 3])

# 尝试调用 tensor.item() 方法，会引发 ValueError 异常
value = tensor.item()

import torch

# 创建一个张量
tensor = torch.tensor([1, 2, 3])

# 使用原地操作函数将张量的所有元素都乘以2
newt = tensor.mul_(2)

# 此时 tensor 的值已经被修改
print(tensor)
print(newt)

# 创建另一个张量并乘以2，但不是原地操作
tensor2 = torch.tensor([1, 2, 3])
new_tensor = tensor2 * 2

# tensor2 的值不会被修改
print(tensor2)

# new_tensor 是一个新的张量，存储乘以2后的结果
print(new_tensor)

import torch
import torch.nn as nn

# 使用 Sequential 定义一个简单的神经网络模型
model = nn.Sequential(
    nn.Linear(10, 20),  # 全连接层，输入大小为10，输出大小为20
    nn.ReLU(),           # ReLU激活函数
    nn.Linear(20, 1)     # 全连接层，输入大小为20，输出大小为1
)
print(model)
print(model.named_parameters)

# 定义输入张量
input_tensor = torch.randn(1, 10)

# 执行前向传播
# output_tensor = model(input_tensor)
output_tensor = model.forward(input_tensor)

print("Input tensor:")
print(input_tensor)
print("\nOutput tensor:")
print(output_tensor)

import torch
import torch.nn as nn

# 使用自定义类继承自 nn.Module 定义神经网络模型
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # 全连接层，输入大小为10，输出大小为20
        self.relu = nn.ReLU()          # ReLU激活函数
        self.fc2 = nn.Linear(20, 1)    # 全连接层，输入大小为20，输出大小为1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建自定义模型的实例
model = CustomModel()
print(model)
print(model.named_parameters)

# 定义输入张量
input_tensor = torch.randn(1, 10)

# 执行前向传播
# output_tensor = model(input_tensor)
output_tensor = model.forward(input_tensor)

print("Input tensor:")
print(input_tensor)
print("\nOutput tensor:")
print(output_tensor)

import torch

# 定义一个张量并设置 requires_grad=True，表示要对其求梯度
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)

# 定义一个计算图
z = x * y + torch.sin(x)

# 自动求导
z.backward()

# 输出 x 和 y 的梯度
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)

import torch

# 定义一个张量并设置 requires_grad=True，表示要对其求梯度
x = torch.tensor(3.0, requires_grad=True)
with torch.no_grad():
  y = torch.tensor(4.0)

# 定义一个计算图
z = x * y + torch.sin(x)

# 自动求导
z.backward()

# 输出 x 和 y 的梯度
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)

import torch
import torch.nn as nn

# 使用 Sequential 定义一个简单的神经网络模型
model = nn.Sequential(
    nn.Linear(10, 20),  # 全连接层，输入大小为10，输出大小为20
    nn.ReLU(),           # ReLU激活函数
    nn.Linear(20, 1)     # 全连接层，输入大小为20，输出大小为1
)

# 定义输入张量
input_tensor = torch.randn(1, 10)

# 执行前向传播
output_tensor = model(input_tensor)

# 创建优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 定义梯度回归为1的目标值
target_value = torch.ones_like(output_tensor)

# 计算损失函数
loss_function = nn.MSELoss()
loss = loss_function(output_tensor, target_value)

# 反向传播计算梯度
optimizer.zero_grad()  # 梯度清零
loss.backward()  # 反向传播
optimizer.step()  # 更新参数

# 限制参数值回归为1
with torch.no_grad():
    for param in model.parameters():
        param.clamp_(1)  # 将参数限制在一个范围内，确保其值为1

print("Model parameters after regression:")
for name, param in model.named_parameters():
    print(name, param.data)

torch.save(model.state_dict(), 'model.pth')

from google.colab import files
import os
print( os.getcwd() )
print( os.listdir() )

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# 准备数据
x = torch.randn(100, 1)
y = 3 * x + 2 + torch.randn(100, 1) #gt

# 实例化模型、损失函数和优化器
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
epochs = 10
losses = []
gradients = []

for epoch in range(epochs):
    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 保存梯度
    gradients.append(model.fc.weight.grad.item())

    # 更新参数
    optimizer.step()

    # 保存损失
    losses.append(loss.item())

# 可视化损失和梯度
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(gradients, label='Gradient')
plt.xlabel('Epoch')
plt.ylabel('Gradient')
plt.title('Gradient')
plt.legend()

plt.show()
```



## 要点

### Q

- 什么是机器学习，包含哪三个基本元素？
- 机器学习可以分为哪几类？ 
- 什么是线性模型？
- 什么是线性回归？
- 线性回归在什么情况下有Closed form的解？什么情况下没有？ 
-  什么是Ridge regression？什么是Lasso regression？他们的解有什么特点？
-  什么是广义的线性模型?

### A

- **机器学习**指算法的设计与分析使得我们能够基于经验提升模型在某些任务上的表现 

- **三要素**：任务、经验、表现

- 机器学习可以分为

  - 有监督学习 
    - 分类问题、回归问题… 

  - 无监督学习 
    - 聚类、密度估计、降维… 

  - 半监督学习 

  - 弱监督学习 

  - 强化学习

- **线性模型**:学习特征X的一种线性组合来进行预测Y

- **线性回归**:给定数据 ,用一个**线性模型**估计最接近真实$y_i$(ground truth )的连续标量Y

  -  ${f}(x_i) = {w}^{T}x_i +b$, such that$ {f}(x_i) \approx{y_i}$

  - $w,b$ 是要学习的模型参数

- 对于$(A^{T}A)\hat{\beta} = A^{T}Y$
  - 如果$A^{T}A$可逆,有$\hat{\beta}=(A^{T}A)^{-1}A^{T}Y$,即$\hat{f}_n^{L}=X\hat{\beta}$,此时有闭合解
  - 当**样本数量<特征维度**的时候,($n<p$),此时无闭合解
- Ridge Regression
  - 与高斯分布相关,对应$L_2$范数,特点是**减小参数的取值**
- Lasso regression
  - 与拉普拉斯分布相关,对应$L_1$​范数,特点是**非零参数更少**
- 广义的线性模型: 我们可以考虑任意⼀种单调可微的函数 $g$使得$y = g^{-1}(w^{T}x+b)$​
