---
title: "从Google Colab到布大超算"
subtitle: ""
date: 2020-07-07T10:33:30+01:00
categories: [Tech, Configuration]
tags: [Supercomputers]
slug: "use-uob-supercomputers"
toc: true
displayCopyright: true
dropCap: false
---

在自己的Windows电脑上使用BlueCrystal Phase 4。

<!--more-->

Google Colab提供的GPU资源我在三到五月间用了几次，用来训练YOLOv3模型，成功训练出来的是两次，分别是训练16张COCO数据集[^1]和700多张的自标注数据集[^2]，前一个好说，数据集小训练得飞快，后一个训练简直是损耗我的耐心，过程中出现了一些印象深刻的坑：

[^1]:https://github.com/JinhangZhu/project-diary/issues/2#issue-579922097
[^2]:https://github.com/JinhangZhu/project-diary/issues/8#issue-612799043

- 允许一次持续使用免费GPU最多12小时。我最多一次用了8小时，虽说没达到12小时，但是我印象深刻，因为数据集再大一倍就不是8小时了，需要加上checkpoint让模型能在下一次免费的12小时内继续训练。
- GPU资源的运算能力并不是很大。就拿我用的[代码](https://github.com/ultralytics/yolov3)来说，作者采用的`batch-size`达到了16或者32，然而我的情况是8就是运算极限了，得继续缩小为4。
- 挂载断开连接。这是一大坑，Colab可能因为各种原因disconnected，比如电脑自动休眠，窗口和鼠标长时间不活跃等。有一次不知道是因为反复重复连接GPU还是怎么的，Colab不让我连了，我只好换了个账号训练，有号任性。解决的办法也有，让电脑尽量插电不息屏，设置浏览器的js自动点击或者使用按键精灵。
- 量大的文件位置得上传。我是上传到Drive然后在Colab里面开头mount Drive，为什么要这么麻烦呢？就是因为Colab的临时空间只有30多G，使用COCO的话先下载压缩包，20G没了，然后解压，解压过程是不能删压缩包的，所以存储必超。另一个问题就是即使数据集小能塞进临时空间，但是重连GPU之后临时空间会消失......网速慢的时候，这个问题会搞人。

但是还是觉得Google很棒，能提供免费的GPU资源给学生党用。后来，RRP老师Richards在我的Research Plan里的关于使用Colab的risk分析处给了个评语：UoB HPC？我当时没搞明白他写啥也没去查，后来跟导师反映计算资源的时候直到了布大给的BlueCrystal Phase 4 (BC4)。直到今天申请的时候才知道就是HPC提供了BC4。布大的[HPC](http://uob-hpc.github.io/)拥有BlueCrystal超级计算机，BlueCrystal Phase 3 (BC3)和BC4都在2013年和2016年的[超算500强](http://www.top500.org/)里，今年没上榜但是要帮忙训练个模型不是绰绰有余？BC4有将近16,000核和64个NVIDIA P100 GPUs，运算速度能达到600万亿次。BC3适用于一般的单处理器和小型的并行运算，BC4就是用来做大型并行运算的，这些术语我并不是很在行，但是加上导师的推荐，想来训练大型CNN模型应该是要用BC4。

> HPC的管理员也是这么说的：
>
> I've created accounts for you on Bluecrystal Phase 3 and 4. Since you'll
> be running computer vision and NN jobs I would recommend using Bluecrystal 4.
>
> The GPU hardware and software stack is much more up to date on there.

<img src="https://live.staticflickr.com/3162/2610243074_6be9589068.jpg" title="Credit: https://www.flickr.com/photos/9123851@N03/2610243074">

下面就介绍一下布大做项目的学生如何申请资格用上曾经的超算500强。

## 配置连接

### 申请资格

首先在 https://www.acrc.bris.ac.uk/acrc/phase4.htm 网页看看大致的介绍，左侧菜单有Application Form，申请表除了需要基本学生个人信息外，还需要Project Code，这个需要找导师要。填好提交后等待消息。邮件回复开通资格后，就可以进行下面的步骤了。

### 安装VPN

N.B. 如果是在校园网Eduroam环境下使用电脑，不需要配置VPN。

https://www.bris.ac.uk/it-services/advice/homeusers/uobonly/uobvpn

选择相应的系统，比如Windows 10，下载Big-IP Edge客户端并安装。从启动页打开客户端，出现下面的界面：

![image-20200708105230994](https://i.loli.net/2020/07/08/9cfEwS4FN67JVrK.png)

我电脑因为配置过下一节的远程桌面，电脑有了我的学生信息，所以直接弹出下面的可直接选择登录的界面。如果没有，就到[这个网页](https://www.bris.ac.uk/it-services/advice/homeusers/uobonly/uobvpn/howto/windows/)跟着步骤走。

![image-20200708105207772](https://i.loli.net/2020/07/08/xiUYtdgVS9DmuKw.png)

连好后，桌面右下角图标就显示了客户端的图标：

![image-20200708105550613](https://i.loli.net/2020/07/08/OwpYU6QyMCHGTos.png)

### 配置远程桌面环境

N.B. 如果是在校园网Eduroam环境下使用电脑，不需要配置远程桌面环境。

http://www.bristol.ac.uk/it-services/advice/homeusers/remote/studentdesktop

一般来说选择第一中connection下载，点击之后输入UoB的账号密码就可以连接上UoB的桌面了。

### 安装PuTTY

N.B. 如果是在远程桌面环境下，不需要安装PuTTY。

https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html，直接选择64-bit的`msi`文件下载安装。

点击PuTTY启动，在主页面的*Host Name*输入：

```
bc4login.acrc.bris.ac.uk
```

![image-20200707131535951](https://i.loli.net/2020/07/07/a8JRvYOVC6Bz7GE.png)

点击*Open*，按照要求输入UoB账号和密码：

![image-20200708110236658](https://i.loli.net/2020/07/08/tdPxLJWOqiQE7B1.png)

## 使用BC4



## References

[3 个相见恨晚的 Google Colaboratory 奇技淫巧！](https://zhuanlan.zhihu.com/p/56581879)

[苦逼学生党的Google Colab使用心得](https://zhuanlan.zhihu.com/p/54389036)

[最新超算 500 强名单：整体算力进入“千万亿次时代”](http://www.mittrchina.com/news/4009)

[High Performance Computing](https://www.bristol.ac.uk/acrc/high-performance-computing/)