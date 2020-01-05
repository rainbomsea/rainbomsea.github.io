---
layout:     post
title:      Tensorflow 编程概念
subtitle:   谷歌机器学习速成课程之Tensorflow 编程概念
date:       2019-05-08
author:     RainbomSea
header-img: img/post-web.jpg
catalog: true
tags:
    - 机器学习
    - 谷歌机器学习快速课程
---

> 仓库代码: [Tensorflow 编程概念](https://github.com/RainbomSea/Jupyter-Notebook/blob/master/%E8%B0%B7%E6%AD%8C%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%80%9F%E6%88%90%E8%AF%BE%E7%A8%8B/TensorFlow%20%E7%BC%96%E7%A8%8B%E6%A6%82%E5%BF%B5.ipynb)

**学习目标**:

* 学习 TensorFlow 编程模型的基础知识，重点了解以下概念：
  * 张量
  * 指令
  * 图
  * 会话
* 构建一个简单的 TensorFlow 程序，使用该程序绘制一个默认图并创建一个运行该图的会话

**注意：**请仔细阅读本教程。TensorFlow 编程模型很可能与您遇到的其他模型不同，因此可能不如您期望的那样直观。

## 概念概览

TensorFlow 的名称源自**张量**，张量是任意维度的数组。借助 TensorFlow，您可以操控具有大量维度的张量。即便如此，在大多数情况下，您会使用以下一个或多个低维张量：

* **标量**是零维数组（零阶张量）。例如，`\'Howdy\'` 或 `5`
* **矢量**是一维数组（一阶张量）。例如，`[2, 3, 5, 7, 11]` 或 `[5]`
* **矩阵**是二维数组（二阶张量）。例如，`[[3.1, 8.2, 5.9][4.3, -2.7, 6.5]]`

TensorFlow **指令**会创建、销毁和操控张量。典型 TensorFlow 程序中的大多数代码行都是指令。

TensorFlow **图**（也称为**计算图**或**数据流图**）是一种图数据结构。很多 TensorFlow 程序由单个图构成，但是 TensorFlow 程序可以选择创建多个图。图的节点是指令；图的边是张量。张量流经图，在每个节点由一个指令操控。一个指令的输出张量通常会变成后续指令的输入张量。TensorFlow 会实现**延迟执行模型**，意味着系统仅会根据相关节点的需求在需要时计算节点。

张量可以作为**常量**或**变量**存储在图中。您可能已经猜到，常量存储的是值不会发生更改的张量，而变量存储的是值会发生更改的张量。不过，您可能没有猜到的是，常量和变量都只是图中的一种指令。常量是始终会返回同一张量值的指令。变量是会返回分配给它的任何张量的指令。

要定义常量，请使用 `tf.constant` 指令，并传入它的值。例如：

```python
  x = tf.constant([5.2])
```

同样，您可以创建如下变量：

```python
  y = tf.Variable([5])
```

或者，您也可以先创建变量，然后再如下所示地分配一个值（注意：您始终需要指定一个默认值）：

```python
  y = tf.Variable([0])
  y = y.assign([5])
```

定义一些常量或变量后，您可以将它们与其他指令（如 `tf.add`）结合使用。在评估 `tf.add` 指令时，它会调用您的 `tf.constant` 或 `tf.Variable` 指令，以获取它们的值，然后返回一个包含这些值之和的新张量。

图必须在 TensorFlow **会话**中运行，会话存储了它所运行的图的状态：

```python
with tf.Session() as sess:
  initialization = tf.global_variables_initializer()
  print(y.eval())
```

在使用 `tf.Variable` 时，您必须在会话开始时调用 `tf.global_variables_initializer`，以明确初始化这些变量，如上所示。

**注意：**会话可以将图分发到多个机器上执行（假设程序在某个分布式计算框架上运行）。有关详情，请参阅[分布式 TensorFlow](https://www.tensorflow.org/deploy/distributed)。

### 总结

TensorFlow 编程本质上是一个两步流程：

1. 将常量、变量和指令整合到一个图中。
2. 在一个会话中评估这些常量、变量和指令。

## 创建一个简单的 TensorFlow 程序

我们来看看如何编写一个将两个常量相加的简单 TensorFlow 程序。

### 添加 import 语句

与几乎所有 Python 程序一样，您首先要添加一些 `import` 语句。
当然，运行 TensorFlow 程序所需的 `import` 语句组合取决于您的程序将要访问的功能。至少，您必须在所有 TensorFlow 程序中添加 `import tensorflow` 语句：

```python
import tensorflow as tf
```

**请勿忘记执行前面的代码块（`import` 语句）。**

其他常见的 import 语句包括：

```python
import matplotlib.pyplot as plt # 数据集可视化。
import numpy as np              # 低级数字 Python 库。
import pandas as pd             # 较高级别的数字 Python 库。
```

TensorFlow 提供了一个**默认图**。不过，我们建议您明确创建自己的 `Graph`，以便跟踪状态（例如，您可能希望在每个单元格中使用一个不同的 `Graph`）。

```python
import tensorflow as tf

# 创建一个图
g = tf.Graph()

# 将图形设置为“默认”图形。
with g.as_default():
    # 组装包含以下三个操作的图形：
    #   * 两个tf.constant操作来创建操作数
    #   * 一个tf.add操作添加两个操作数。
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    sum = tf.add(x, y, name="x_y_sum")
    # 现在创建一个回话
    # 这个回话将运行默认图
    with tf.Session() as sess:
        print(sum.eval())
```
