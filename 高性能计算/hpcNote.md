---
title: 高性能计算
date: 2024-03-28 17:49:28
tags: 高性能计算
index_img: 
banner_img: 
description：
---



## Profiling

- 阿姆达尔定律(优化热点的收益最大!!)
- profiling就是找到这个热点
  - 找出程序中那部分耗时最长(宏观层面)
  - 获取某个函数运行的Metric,以优化具体的函数(微观层面)

- Profiler
  - 一种辅助工具用于分析程序运行的情况
  - 间歇性地打断程序运行,同时采样当前正在执行的指令:可以知道那些指令占用了大部分时间
  - ……

```python
import tensorflow as tf


tensorflow_version = tf.__version__
gpu_available = tf.test.is_gpu_available()

print("tensorflow version:",tensorflow_version,"\tGPU available",gpu_available)

a = tf.constant([1.0,2.0],name = "a")
b = tf.constant([1.0,2.0],name = "b")
result = tf.add(a,b,name = "add")
print(result)
```
## MPI


`mpi.h`是MPI（Message Passing Interface）库的C头文件，它提供了一组用于在不同进程之间传递消息的函数，允许在多个计算机上运行的程序协作解决问题。这是并行计算中常见的一种方法，特别是在高性能计算(HPC)领域。MPI提供了一种方便的方式来编写可以在多个处理器上并行运行的程序。

下面是一些基本的`mpi.h`用法，以及在您提供的代码片段中出现的MPI函数：

1. 初始化和终止MPI：
   - `MPI_Init(&argc, &argv);`：初始化MPI环境，这个函数必须是MPI程序中第一个被调用的MPI函数。它接受main函数的`argc`和`argv`作为参数，这样可以让MPI处理任何MPI库特定的命令行参数。
   - `MPI_Finalize();`：终止MPI环境，清理所有MPI状态。在程序结束前，这个函数应该是最后调用的MPI函数。

2. 获取进程信息：
   - `MPI_Comm_rank(MPI_COMM_WORLD, &rank);`：获取当前进程的秩（`rank`）。在一个MPI程序中，每个进程都被分配一个唯一的秩，用于标识。`MPI_COMM_WORLD`是所有进程的默认通信器（communicator）。
   - `MPI_Comm_size(MPI_COMM_WORLD, &nprocs);`：获取在给定通信器（这里是`MPI_COMM_WORLD`）中的进程总数。

3. 发送和接收消息（在您提供的代码片段中没有这部分，但这是MPI的核心功能之一）：
   - `MPI_Send(void* data, int count, MPI_Datatype datatype, int destination, int tag, MPI_Comm communicator);`：发送消息到指定的目的地进程。
   - `MPI_Recv(void* data, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm communicator, MPI_Status *status);`：从指定的源进程接收消息。

4. 并行计算控制：
   - `MPI_Abort(MPI_COMM_WORLD, 1);`：在出现错误的情况下，终止所有与`MPI_COMM_WORLD`通信器关联的进程。第二个参数是错误代码。

MPI程序通常遵循以下模式：
- 初始化MPI。
- 获取当前进程的秩和总进程数。
- 根据进程的秩，执行不同的任务（通常秩为0的进程执行主要协调任务）。
- 使用MPI的通信功能在进程间分发任务和收集结果。
- 最终，完成所有并行任务后，终止MPI环境。

MPI是一个强大的库，用于分布式内存系统上的并行编程，它支持点对点通信和集合通信，提供了一组丰富的并行编程工具。
