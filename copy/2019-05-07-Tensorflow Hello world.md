---
layout:     post
title:      Tensorflow Hello World
subtitle:   谷歌机器学习速成课程之Tensorflow Hello World
date:       2019-05-07
author:     RainbomSea
header-img: img/post-web.jpg
catalog: true
tags:
    - 机器学习
    - 谷歌机器学习快速课程
---

> 仓库代码: [Tensorflow Hello World](https://github.com/RainbomSea/Jupyter-Notebook/blob/master/谷歌机器学习速成课程/Tensorflow%20%20Hello%20World.ipynb)

**学习目标：** 在浏览器中运行 TensorFlow 程序。

以下代码块为“Hello World”TensorFlow 程序。

其中包含初始化代码（导入 TensorFlow 模块并启用“eager execution”，我们将在后续练习中详细介绍此操作），然后输出“Hello, world!”字符串常量。

```python
import tensorflow as tf
try:
    tf.contrib.eager.enable_eager_execution()
except ValueError:
    pass

tensor = tf.constant('Hello world!')
tensor_value = tensor.numpy()
print(tensor_value)
```
