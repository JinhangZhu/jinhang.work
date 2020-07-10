---
title: "NNDL-1：使用神经网络识别手写数字"
date: 2020-05-16T22:17:03+01:00
categories: [Tech,"Machine learning"]
tags: [Neural networks, Deep learning, MNIST]
slug: "nndl-learn-nn-handwritten-digits"
---

神经网络与机器学习一：使用神经网络识别手写数字。本文目的：梯度下降的数学基础和基于Python3的简单前馈神经网络。代码：<a class="github-button" href="https://github.com/jinhangzhu/nndl-book" data-icon="octicon-star" aria-label="Star jinhangzhu/nndl-book on GitHub">Star</a>

<!--more-->


<!-- TOC -->

- [关于这本书](#关于这本书)
  - [以原理为导向](#以原理为导向)
  - [动手实践](#动手实践)
- [使用神经网络识别手写数字](#使用神经网络识别手写数字)
  - [感知器 （Perceptron）](#感知器-perceptron)
  - [S型神经元（Sigmoid neurons）](#s型神经元sigmoid-neurons)
  - [神经网络的架构](#神经网络的架构)
  - [简单的手写数字分类网络](#简单的手写数字分类网络)
  - [使用梯度下降算法进行学习](#使用梯度下降算法进行学习)
    - [数学基础](#数学基础)
    - [运用梯度下降](#运用梯度下降)
  - [实现网络](#实现网络)
  - [效果反思](#效果反思)
- [总结](#总结)

<!-- /TOC -->


<a id="markdown-关于这本书" name="关于这本书"></a>

## 关于这本书

<a id="markdown-以原理为导向" name="以原理为导向"></a>

### 以原理为导向

- 理解核心理念
- 而非围绕特定的程序库

> 如果你想理解神经⽹络中究竟发⽣了什么，如果你想要了解今后⼏年都不会过时的原理，那么只是学习些热⻔的程序库是不够的。你需要领悟让神经⽹络⼯作的原理。技术来来去去，但原理是永恒的。

<a id="markdown-动手实践" name="动手实践"></a>

### 动手实践

编程：书本使用Python 2.7开发小型神经网络库，代码：https://github.com/mnielsen/neural-networks-and-deep-learning。

数学：包含必要的数学细节。

建议：完成大多数的练习，不建议做完所有的项目，找到适合自己的/自己关心的。

<a id="markdown-使用神经网络识别手写数字" name="使用神经网络识别手写数字"></a>

## 使用神经网络识别手写数字

神经网络不同于传统的建立数学物理模型解决问题，而是使用训练样本来训练出一套可以对样本进行预测的规则系统。

![image-20200516132117732](https://i.loli.net/2020/05/17/aujXsNkUoMO7YWG.png)

采用识别手写数字的原因是：

- 具有挑战性：不同于传统建模
- 不需要复杂的算法和运算。

> 当然，如果仅仅为了编写⼀个计算机程序来识别⼿写数字，本章的内容可以简短很多！但前进的道路上，我们将扩展出很多关于神经⽹络的关键的思想，其中包括两个重要的⼈⼯神经元（感知器和S 型神经元），以及标准的神经⽹络学习算法，即随机梯度下降算法。

<a id="markdown-感知器-perceptron" name="感知器-perceptron"></a>

### 感知器 （Perceptron）

一个`感知器`接受几个二进制输入，并且产生一个二进制输出：

<img src="https://i.loli.net/2020/05/17/Vri9C8a2XNcUmdK.png" alt="image-20200516152341141" style="zoom:80%;" />

发明者Frank Rosenblatt引入了`权重`来表示相应输入的重要性。神经元的输出（0/1）由加权和$\sum_jw_jx_j$和一些`阈值`决定。感知器就是把输入做加权和然后按照阈值判断输出的人工神经元。不过在之后，我们不再局限于感知器的定义，会将这些人工神经元视作包含带值的单元，不一定都有输入和输出。

![image-20200516152920351](https://i.loli.net/2020/05/17/Anqgj3etYJVhyba.png)

对于一个稍微复杂的感知器网络（MLP），第一层感知器通过权衡输入做出三个非常简单的决定；而第二层的感知器就可以做出比第一层做出的更复杂和抽象的决策；层数越深，相应的感知器就可以做出更复杂的决策。

我们用`偏置`$b=-threshold$来代替阈值，用向量点乘代替求和符号，如下式。偏置决定了感知器输出1有多容易。
$$
output =
\begin{cases}
0,  & \text{if $w\cdot x + b \leq0$} \\
1, & \text{if $w\cdot x + b > 0$}
\end{cases}
$$
运用感知器，我们可以实现逻辑功能，只需要规定权重和偏置。比如实现一个与非门，权重均为-2，偏置为3：

<img src="https://i.loli.net/2020/05/17/9buZ7K2qNsnPr3i.png" alt="image-20200516154323502" style="zoom:80%;" />

对于输入00，$(-2)\times0+(-2)\times0+3=3>0\rightarrow 1$；对于输入11，$(-2)\times 1+(-2)\times 1+3=-1\leq0\rightarrow0$。

<a id="markdown-s型神经元sigmoid-neurons" name="s型神经元sigmoid-neurons"></a>

### S型神经元（Sigmoid neurons）

权重或偏置的微小改变可以引起输出的微小改变（$w+\Delta w\rightarrow output+\Delta output$），那么我们可以利用此事实修改这两类参数。为什么引入S型神经元呢？因为感知器是线性的，参数的微笑改变有时候会引起输出的完全反转，对于某个分类的适应可能导致对于另一分类的输出大幅度改变。S型可以更缓和这种微小变化。

同样的，S型神经元有多输入，输入时0到1中任意值，原本的感知器输出是$w\cdot x+b$，现在是$\sigma(w\cdot x+b)$，$\sigma$函数也叫作`逻辑函数`（标准形式）。
$$
\begin{aligned}
\sigma(z)
&=\frac{1}{1+e^{-z}}\\\\
&=\frac{1}{2}+\frac{1}{2}\cdot tanh(\frac{z}{2})
\end{aligned}
$$
逻辑函数就是将所有的$(-\infty,+\infty)$映射到$[0,1]$上。之前的z的函数，相当于是阶跃函数。

<img src="https://i.loli.net/2020/05/17/rqpXHE5cel84tIU.png" alt="image-20200516155948524" style="zoom:80%;" />

逻辑函数更平滑意味着权重和偏置的微小变化。微积分说明：

<img src="https://i.loli.net/2020/05/17/E7pCH3xJoZeQrXL.png" alt="image-20200516161331025" style="zoom:80%;" />

在后面，逻辑函数还会用来做`激活函数`，它会简化数学计算，因为指数在求导的时候很让人愉悦（但注意最大值是1/4，所以多个$\sigma$函数导数相乘导数会慢慢消失）：
$$
\frac{d(\sigma(x))}{x}=\sigma(x)\cdot (1-\sigma(x))
$$

> 练习：
>
> - 证明所有权重和偏置乘以正常数c时，网络的行为不变。
>
>   从导数角度思考，最后的结果是导数最大值乘以了c，但是偶函数性质不变。
>
> - 证明在假设$w\cdot x+b\neq0, \forall x$成立时，在正常量$c\rightarrow \infty$情况下，S型神经元的行为和感知器网络完全一致。
>
>   当$w\cdot x+b\neq0$时，我们可以保证S型神经元在z输出为0上下，能够保证唯一输出，而且根据上一问，c无穷大，导数最大值为无穷大，刚好时阶跃函数的导数：冲激函数；要是假设不满足，S型神经元可以保证$z=0$时输出为0.5，而感知器输出就是1了（按照规定）。

<a id="markdown-神经网络的架构" name="神经网络的架构"></a>

### 神经网络的架构

<img src="https://i.loli.net/2020/05/17/oNQPXETV2OkKzJx.png" alt="image-20200516163552528" style="zoom:80%;" />

`输入层-隐藏层-输出层`，历史上称为多层感知器（MLP）。

> 关于隐藏层设计：
>
> 相⽐于神经⽹络中输⼊输出层的直观设计，隐藏层的设计则堪称⼀⻔艺术。特别是，通过⼀些简单的经验法则来总结隐藏层的设计流程是不可⾏的。相反，神经⽹络的研究⼈员已经为隐藏层开发了许多设计最优法则，这有助于⽹络的⾏为能符合⼈们期望的那样。例如，这些法则可以⽤于帮助权衡隐藏层数量和训练⽹络所需的时间开销。在本书后⾯我们会碰到⼏个这样的设计最优法则。

这样的叫前馈网络（feedforward net），不同于存在反馈环路的递归神经网络（Recurrent Neural Network），RNN会保存部分神经元的激活状态，将其输出在一段时间后作为部分神经元的输出，所以一般用来识别时间序列的特征，但是这本书只考虑前馈型的。

<a id="markdown-简单的手写数字分类网络" name="简单的手写数字分类网络"></a>

### 简单的手写数字分类网络

专注于一次分类一个单独的数字，如：<img src="https://i.loli.net/2020/05/17/vdnVJo1xKPC7eyY.png" alt="image-20200516164334125" style="zoom:50%;" />而不是：<img src="https://i.loli.net/2020/05/17/g35OrJAhBXFWS7x.png" alt="image-20200516164405891" style="zoom:50%;" />。不考虑数字序列的分割问题，如果考虑，可以思考这样的方法：数字分类器对每一个切分片段打分，要是每一个片段的置信度都比较高，那么这种分割方式就能得到较高分数。对于手写数字分类的网络：

<img src="https://i.loli.net/2020/05/17/4JsxLO6TjnhHYRw.png" alt="image-20200516164703703" style="zoom:80%;" />

输入层的728来自$28\times 28$，是输入图像的size，输入像素为灰度级，0.0表示白色，1.0表示黑色。隐藏层n=15是一个示例，输出层10个神经元对应十个数字。实验证明，十个输出神经元的网络比基于二进制码的四个输出神经元的网络更好。把二进制的位和数字形状联系起来并没有什么关系，而每个数字都有明显的形状特征，全都按照独立的来看更好找到起主要作用的局部形状。

> 练习：
>
> 在上面的三层网络后额外加一层四个神经元的输出层表示二进制的数位，如图。现在需要给输出层寻找一些合适的权重和偏置。（假设原三层网络在第三层得到正确输出的激活值至少是0.99，得到错误的输出的激活值至多是0.01）.
>
> <img src="https://i.loli.net/2020/05/17/I7RKTNOEjtaDUWv.png" alt="image-20200516171147847" style="zoom:80%;" />
>
> 思考每一个数字对数位的贡献，然后希望z值能用0区分就行。比如从$a_0^3$（第三层第一个，表示数字0的置信度）到第四层的所有权重从高位到低位依次为$[-100,-100,-100,-100]$，0的输出给每个输出神经元都导致了很小的负数z，偏置取很小的正数就行，就可以让阶跃函数都输出0。

<a id="markdown-使用梯度下降算法进行学习" name="使用梯度下降算法进行学习"></a>

### 使用梯度下降算法进行学习

MNIST数据分为训练集：60000张$28\times 28$图像，测试集：10000张$28\times 28$图像。网络输入$x: 28\times 28=784$的向量，输出$y(x)$：10维列向量。

`代价函数`（损失/目标）：
$$
C(w,b)\equiv\frac{1}{2n}\sum_x\|y(x)-a\|^2
$$
$C$为`二次`代价函数，也叫`均方误差`或者`MSE`。为什么用二次？相对平滑，容易利用微小改变来最小化代价。为什么不用其他平滑的？**之后再说**。

<a id="markdown-数学基础" name="数学基础"></a>

#### 数学基础

一般最小化问题的方法是微积分算极值点，但是神经网络有大量的变量，导致代价函数会及其复杂，这时引入梯度下降。对于简单的代价函数$C(v)$，在$v_1,v_2$方向分别移动很小的量，$C$的变化为：

<img src="https://i.loli.net/2020/05/17/rpdIbt23ulwo7Vj.png" alt="image-20200516173211505" style="zoom:80%;" />

需要让$\Delta C$为负，才能让代价下降，定义$\Delta v\equiv (\Delta v_1, \Delta v_2)^T$为$v$i变化的向量。那么$C$的梯度就是偏导数的向量——梯度向量：
$$
\nabla C\equiv(\frac{\partial C}{\partial v_1},\frac{\partial C}{\partial v_2})^T
$$
所以变化量表示为
$$
\Delta C\approx\nabla C \cdot \Delta v
$$
代价的梯度向量把$v$的变化关联为代价的变化，那么得到一个`负`变化就很简单了，找到一个$v$变化的方向，等效为乘以一个负号和正的`学习效率`。
$$
\Delta v = - \eta \nabla C\\\\
\Delta C\approx\nabla C \cdot \Delta v=\eta \|\nabla C\|^2
$$
那么新的$v'\gets v-\eta \nabla C$。可以证明，是的$\Delta C$取得最小值的$\Delta v=\eta \nabla C$，所以说梯度下降法可被视为一种**在$C$下降最快方向上做微小改变**的方法。

> 练习：
>
> - 证明取得最小值条件。
>
>   采用可惜-施瓦茨不等式，两个向量内积取最值当然是线性相关的时候。
>
> - 如果代价函数时一元函数呢？如何用几何解释？
>
>   此时假设自变量为x，那么x的变化就是步长，视作定值，找最小值的过程简化为$C$从出发点开始减小到最近的极小值的过程。

<a id="markdown-运用梯度下降" name="运用梯度下降"></a>

#### 运用梯度下降

$$
w_k \gets w_k-\eta\frac{\partial C}{\partial w_k}\\\\
b_l \gets b_l-\eta\frac{\partial C}{\partial b_l}
$$

传统的`梯度下降`是将所有训练样本单独计算了梯度值然后取平均值，一般很慢。`随机梯度下降（SGD）`每次选取小量$m$个样本来计算梯度值，这样可以加速学习。

SGD每次随机选取的训练样本可以标记为$X_1, X_2,...,X_m$，叫做小批量数据（mini-batch）。假设m足够大，期望这个批数据的平均梯度应该大致等于全体样本的平均梯度。所以可以用mini-batch的梯度来`估算`整体梯度。

<img src="https://i.loli.net/2020/05/17/rBDH3G7XLP8MeWy.png" alt="image-20200516180956524" style="zoom:80%;" />

> ⼈们有时候忽略1/n，直接取单个训练样本的代价总和，⽽不是取平均值。这对我们不能提前知道训练数据数量的情况下特别有效。例如，这可能发⽣在有更多的训练数据是实时产⽣的情况下。同样，⼩批量数据的更新规则有时也会舍弃前⾯的1/m。

SGD中每一个迭代期（epoch）进行一次参数更新：

<img src="https://i.loli.net/2020/05/17/jdag49Fhe1ypvA6.png" alt="image-20200516181136013" style="zoom:80%;" />

> 练习：
>
> 极端的SGD将m取1，选取一个样本输入，更新一次参数，这个过程为在线（online/递增）学习，对比如m=20的SGD，递增学习一对优缺点？
>
> - 优：计算消耗小；
> - 缺：可能难以收敛，因为单个样本的代价梯度不容易用来估算整体梯度。
>
> 深度学习中的batch的大小对学习效果有何影响？ - 程引的回答 - 知乎 https://www.zhihu.com/question/32673260/answer/71137399

<a id="markdown-实现网络" name="实现网络"></a>

### 实现网络

`核心代码`规定了这个神经网络的特性，即神经元层数，层数的分配，每一层的神经元个数，权重，偏置。

```python
class Network:
    def __init__(self, sizes):
        """
        sizes: [2, 3, 1] -> 2 neurons in input layer, 3 neurons in hidden layer, 1 neuron in output layer
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(j, i) for i, j in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(j, 1) for j in self.sizes[1:]]
```

这里采用的权重和偏置是按照均值为0，标准差为1的`高斯分布随机初始化`的，之后也有更好的初始化方法。权重矩阵行数是后一层神经元数，列数是当前层的神经元数，偏置矩阵为后一层神经元数-by-1。这样的顺序方便进行$\sigma$函数的`向量化（vectorization）`：
$$
a'=\sigma(wa+b)
$$
$a'$, $a$, $w$, $b$均为向量。

添加前向传递的方法，给定NN输入，返回NN输出：

```python
    def forward(self, a):
        for weight, bias in zip(self.weights, self.biases):
            a = sigmoid(np.dot(weight, a) + bias)
        return a
```

然后需要实现SGD方法。每一个epoch将打乱的训练集按照mini-batch size分成若干份，每一份batch进行一次iteration，按照`反向传播（Backpropagation）`获得上面说过的代价对权重/偏置的偏导数——梯度向量$\nabla C$。然后按照相应梯度下降公式对权重和偏置进行更新。SGD大部分工作是反向传播在做，但是下章再讲。

```python
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for t in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:  # Update parameters
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Epoch {}: {} / {}".format(t, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(t))

    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate.
        """
        # Store the sum of all samples' nablas
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            sample_nabla_b, sample_nabla_w = self.backprop(x, y)
            # Sum update
            nabla_b = [nb + snb for nb, snb in zip(nabla_b, sample_nabla_b)]
            nabla_w = [nw + snw for nw, snw in zip(nabla_w, sample_nabla_w)]
            self.weights = [w - learning_rate / len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - learning_rate / len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]

```

使用新版的[mnist_loader](https://github.com/MichalDanielDobrzanski/DeepLearningPython35/blob/master/mnist_loader.py)用来导入数据集。

所有代码就位后，命令行中进行训练：

```python
>>> import mnist_loader
>>> training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
>>> import network
>>> net = network.Network([784, 30, 10])
>>> net.SGD(training_data, 30, 10, 3.0, test_data=test_data) # 30次迭代期，批数据大小为10，学习速率3.0
```

训练结果：

```bash
Epoch 0: 7631 / 10000
Epoch 1: 8440 / 10000
Epoch 2: 8564 / 10000
...
Epoch 27: 8882 / 10000
Epoch 28: 9153 / 10000
Epoch 29: 9140 / 10000
```

表现没有书中的（95%）好，是因为**作者使⽤了（不同的）随机权重和偏置来初始化他们的⽹络，采⽤了三次运⾏中的最优结果作为本章的结果。**（我想也使用了**随机种子**）我也可以继续调试各项参数，这里不展开。作者说明了神经网络的效果如果和差的基线测试作比较就很强：如瞎猜（10%），根据图像整体明暗度猜（22.5%）。

<a id="markdown-效果反思" name="效果反思"></a>

### 效果反思

调试神经网络是一项具有挑战的工作。之后的学习里要注意几个问题：

- 学习效率低了还是高了？
- 是否用了很差的初始权重或偏置？
- 训练数据是不是不够？
- 迭代期是否不够？
- 这种结构的神经网络，用来做这个任务是否可能？
- ...

**之后会谈到启发式的方法来选择好的超参数和好的结构**。

> 练习：
>
> 尝试仅有两层的网络[784, 10]，能达到多少识别率？
>
> ```bash
> Epoch 27: 6494 / 10000
> Epoch 28: 6959 / 10000
> Epoch 29: 7019 / 10000
> Training completed in 23 seconds
> ```
>
> 虽然学得更快，但是毕竟学的特征比较浅显，不够丰富，识别率不高。

<a id="markdown-总结" name="总结"></a>

## 总结

最后作者也谈到了一个最著名的支持向量机（SVM）算法，采用`scikit-learn`的API，也可以达到类似神经网络的效果，但是目前最好的记录时神经网络创造的9979/10000 （2013）。这篇达到SOTA效果的论文所用的神经网络也只涉及到简单的算法。作者指出：

<center>


复杂的算法 $\leq$简单的学习算法+好的训练数据

</center><br>

至此，我的部分目的达到了，就是**重新梳理神经网络梯度下降的数学基础**。这篇文章回顾了梯度下降的数学表达，尤其是学习速率$\eta$为正实数来自于要满足负的代价变化量$\Delta C$，参数变化量$\Delta v=-\eta \nabla C$来自于对柯西-施瓦茨不等式的证明（要让代价下降最快）。然后**使用Python3设计了简单的前馈神经网络**，训练方法为小批量数据的随机梯度下降（mini-batch stochastic gradient descent）。小批量数据的训练和传统的全数据集训练不同就在于用批数据的梯度来近似全数据集的梯度，这样的近似值是通过取批数据每个样本的代价梯度的平均值得到的。当然，具体如何得到每个样本的代价梯度$\nabla C_x$，就看下回反向传播分解。