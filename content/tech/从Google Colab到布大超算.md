---
title: "从Google Colab到布大超算"
subtitle: ""
date: 2020-07-07T10:33:30+01:00
categories: [Tech, Configuration]
tags: [Supercomputers]
slug: "use-uob-supercomputers"
toc: true
displayCopyright: true
dropCap: true
---

Google Colab终究是一堆槽点，布大提供了超级计算机BlueCrystal Phase 4来服务研究人员的计算任务，本文就详细讲述如何在自己的电脑上使用这个珍贵资源。注：文章围绕Windows展开，不过差别不大，其他机型可以参考链接和引用解决。

<!--more-->

Google Colab提供的GPU资源我在三到五月间用了几次，用来训练YOLOv3模型，成功训练出来的是两次，分别是训练16张COCO数据集[^1]和700多张的自标注数据集[^2]，前一个好说，数据集小训练得飞快，后一个训练简直是损耗我的耐心，过程中出现了一些印象深刻的坑：

[^1]:https://github.com/JinhangZhu/project-diary/issues/2#issue-579922097
[^2]:https://github.com/JinhangZhu/project-diary/issues/8#issue-612799043

- 允许一次持续使用免费GPU最多12小时。我最多一次用了8小时，虽说没达到12小时，但是我印象深刻，因为数据集再大一倍就不是8小时了，需要加上checkpoint让模型能在下一次免费的12小时内继续训练。

  [Updated 2020.7.17]昨天到今天用Colab训练了自己的数据集，8000+的训练图片，同时也包括每一个epoch的验证，batch-size为8，一共100个迭代期，结果12个小时用完只训练到第19迭代期，这种时候可以采取一些小trick来解决[^trick]。

  [^trick]:[使用pytorch时，训练集数据太多达到上千万张，Dataloader加载很慢怎么办?](https://www.zhihu.com/question/356829360)

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

官方文档：[ Blue Crystal Phase 4 User Documentation](https://www.acrc.bris.ac.uk/protected/bc4-docs/index.html)

### 了解BC4存储

[Link](https://www.acrc.bris.ac.uk/protected/bc4-docs/storage/index.html#bluecrystal-phase-4-storage)。包含`Home Directories`和`Scratch Space`两种空间，前者是固定的20G，后者是512G，用于大型数据集的输入输出，后者路径为`/mnt/storage/scratch`。..这些控件都不会备份..。下面探索一下存储空间，会用到Linux命令[^3]（我当年怎么就没好好学呢），BC4的Linux shell默认是常见[^4]的Bash（在申请资格的时候选择）。

[^3]:可以参考开源工作者撰写的《快乐的Linux命令行》- http://billie66.github.io/TLCL/
[^4]:不懂shell别说你会linux - jay的文章 - 知乎 https://zhuanlan.zhihu.com/p/104729643

---

第一眼注意到的是shell提示符(shell prompt)，意为shell准备好了去接受命令的输入。

```shell
[lm19073@bc4login3 ~]$
```

不多废话，先走第一步， `Hello world!`。用到`echo`命令，首先使用命令帮助[^5]查看命令：

[^5]:学会使用命令帮助 - https://linuxtools-rst.readthedocs.io/zh_CN/latest/base/01_use_man.html

```shell
[lm19073@bc4login3 ~]$ whatis echo
echo (1)             - display a line of text
echo (3x)            - curses input options
```

多的不说，尝试say hello：

```shell
[lm19073@bc4login3 ~]$ echo "Hello world!"
-bash: !": event not found
[lm19073@bc4login3 ~]$ echo 'Hello world!'
Hello world!
[lm19073@bc4login3 ~]$ echo Hello!
Hello!
[lm19073@bc4login3 ~]$ echo '$(date)'
$(date)
[lm19073@bc4login3 ~]$ echo "$(date)"
“▒Tue Jul 14 20:17:27 BST 2020
```

入门成功！现在来看看一下文件系统。基础的三个命令是`pwd`，`ls`，`cd`，分别是Printing working directory, list (out), change directory。其中单纯的`cd`可以直接改回Home directory。

```shell
[lm19073@bc4login3 ~]$ pwd
/mnt/storage/home/lm19073
[lm19073@bc4login3 ~]$ ls
Use  yolov3
[lm19073@bc4login3 ~]$ cd yolov3
[lm19073@bc4login3 yolov3]$ cd
[lm19073@bc4login3 ~]$ pwd
/mnt/storage/home/lm19073
[lm19073@bc4login3 ~]$ cd yolov3
[lm19073@bc4login3 yolov3]$ ls
Dockerfile  README.md  data       models.py         test.py   tutorial.ipynb  weights
LICENSE     cfg        detect.py  requirements.txt  train.py  utils
[lm19073@bc4login3 yolov3]$ cd data
[lm19073@bc4login3 data]$ pwd
/mnt/storage/home/lm19073/yolov3/data
[lm19073@bc4login3 data]$ cd ..
[lm19073@bc4login3 yolov3]$ cd data
[lm19073@bc4login3 data]$ cd
[lm19073@bc4login3 ~]$ pwd
/mnt/storage/home/lm19073
```

要是在我的Home directory再往前一级呢？好像发现了什么不得了的东西，原来大家的都是公开的？以后用`rm`命令可得注意了，公共场所别把人家的删了。

> ![image-20200715001501009](https://i.loli.net/2020/07/15/szvkm1HBaMxRF6Z.png)

```shell
[lm19073@bc4login3 home]$ pwd
/mnt/storage/home
[lm19073@bc4login3 home]$ ls
NOT.lm14358     cl14975.tar.gz    hs12248         madjl           rw15131.tar.gz
UoB             cl15341           hs12828         marrk           rw15164
aa16169         cl15540           hs14458         mb16066         rw15911
...
```

现在检查一下我的存储容量：

```shell
[lm19073@bc4login3 ~]$ mmlsquota --block-size auto
```

![image-20200714203118026](https://i.loli.net/2020/07/15/o2kGhyQv6XupfrW.png)

使用量在`blocks`一栏，总容量在`quota`一栏。根据文档，容量采用软限制，超量将`limit`一栏修改。所以大概就能猜到home 存储空间就是这样按照每个人的ID分开的，每个人的Home directory有20G可用，BC4会监控存储空间大小，一旦超过上限，是通过一个标志位控制是否还有资格继续储存文件。

我需要在BC4上跑NN模型，那么源代码就存储在home directory里面儿，数据集ego-hand不过3G多，不会超过，所以是不用放在scratch space里面的，但是scratch space的设计就是给数据集的IO的，我猜应该把数据放那。然后在Shell中运行代码，调用scratch space的IO。如果说存储空间类型相似或者本身就是同一磁盘分区，我觉得IO读写速度应该不慢。

既然Home directories是很多个用户所使用的，那么scratch space也应该是以用户分的，按照文档，我们进入scratch space看看有什么：

```shell
[lm19073@bc4login3 storage]$ cd scratch/
[lm19073@bc4login3 scratch]$ ls
YP19290         dw14986.tar.gz  jt17776         rb14427.tar.gz
aa16169         dw16383.list    jt18607         rb16536
...
[lm19073@bc4login3 scratch]$ file lm19073/
lm19073/: directory
```

确实如此，而且各自的空间基本都是以文件夹的形式存在。我的目录的长格式为：

```shell
drwxrwx---    2 lm19073 emat19t           4096 Jul  8 09:05 lm19073
```

![image-20200715001227772](https://i.loli.net/2020/07/15/ulcjqhLUzSCyNmH.png)

```shell
[lm19073@bc4login3 lm19073]$ pwd
/mnt/storage/scratch/lm19073
[lm19073@bc4login3 lm19073]$ cd
[lm19073@bc4login3 ~]$ pwd
/mnt/storage/home/lm19073
```

咱发现个什么问题呢？我们一般默认进入Home directory，但是Scratch space和Home directory的路径不一样，所以Linux就有一种叫做软链接的符号链接(symlink)让我们通过访问它来访问软链接指向的真正位置。按照文档的方法创建软链接：

```shell
[lm19073@bc4login3 ~]$ ln -s /mnt/storage/scratch/$USER scratch
[lm19073@bc4login3 ~]$ ls
Use  scratch  yolov3
[lm19073@bc4login3 ~]$ ls -l
total 0
-rw-r--r-- 1 lm19073 emat19t    0 Jul 14 19:20 Use
lrwxrwxrwx 1 lm19073 emat19t   28 Jul 15 00:23 scratch -> /mnt/storage/scratch/lm19073
drwxr-xr-x 8 lm19073 emat19t 4096 Jul 14 19:34 yolov3
```

非常好！scratch这个软链接指向了我的独享的scratch space。这样的一个类似“指针”的操作有什么用呢？比如目标目录`/mnt/storage/scratch/lm19073`所属用户组`emat19t`经常对所管理的目录进行更新，把目录都用版本号来区别，而不是我现在的学号，这就很蛋疼了，每次访问目标目录还得去专门看一下版本号，然后敲出来，不如拿一个reference直接建立访问的桥梁。

### 模块信息

[Link](https://www.acrc.bris.ac.uk/protected/bc4-docs/software/index.html)。我们可以用`module avail`来查看所有模块，`which <modulename>`查看模块位置等，了解了基本信息之后就可以跑代码了。

### 配置舒服的环境

我重连之后发现上回创建的scratch软链接和本身之前clone的yolov3的repo文件夹都还在home目录下，说明这里的空间都可以保留文件（无备份）。

```shell
# 发现基本的模块都有，pip暂无
[lm19073@bc4login3 ~]$ which python3
/usr/bin/python3
[lm19073@bc4login3 ~]$ which git
/usr/bin/git
[lm19073@bc4login3 ~]$ which pip
/usr/bin/which: no pip in (/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/mnt/storage/home/lm19073/.dotnet/tools:/usr/lpp/mmfs/bin:/opt/ddn/ime/bin:/mnt/storage/home/lm19073/.local/bin:/mnt/storage/home/lm19073/bin)
```

我先把yolov3文件夹删掉，使用`rm -rf yolov3`命令。然后重新克隆yolov3：

```shell
[lm19073@bc4login3 ~]$ git clone https://github.com/ultralytics/yolov3
Cloning into 'yolov3'...
...
```

进入scratch space，将COCO2014下载在scratch space里：
```shell
[lm19073@bc4login3 scratch]$ bash /mnt/storage/home/lm19073/yolov3/data/get_coco2014.sh
```

![image-20200715134254844](https://i.loli.net/2020/07/15/5W9VHsBfdqNQAhC.png)

> 这里使用的是脚本`.sh`，通过命令访问网络链接下载数据解压到服务器上。那么<mark>我们自己的数据怎么上传上去呢？</mark>新版文档没有说明，所以我参考了一个较老版本的文档[^6]，对于使用Windows的我来说，下载**winSCP**来管理文件流。

[^6]: How to copy files to and from the cluster - ACRC: BlueCrystal User Guide - https://www.acrc.bris.ac.uk/acrc/pdf/bc-user-guide.pdf

在作者的配置里coco和yolov3目录并列的，为了不修改运行代码的options，我们给`scratch/coco`设置一个和`yolov3`并列的软链接。

<img src="https://i.loli.net/2020/07/16/tzomkrQAecIf5gB.png" alt="image-20200715192838567" title="纯属偶然发现不加软链接名字自动分配basename">

现在环境都配置好了！如果有报错，就参考下一章节Issues来解决。一切就绪后，跑代码。

```shell
[lm19073@bc4login3 yolov3]$ python detect.py
Namespace(agnostic_nms=False, augment=False, cfg='cfg/yolov3-spp.cfg', classes=None, conf_thres=0.3, device='', fourcc='mp4v', half=False, img_size=512, iou_thres=0.6, names='data/coco.names', output='output', save_txt=False, source='data/samples', view_img=False, weights='weights/yolov3-spp-ultralytics.pt')
Using CPU

Model Summary: 225 layers, 6.29987e+07 parameters, 6.29987e+07 gradients
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   1797      0 --:--:-- --:--:-- --:--:--  1789
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100  240M    0  240M    0     0  38.9M      0 --:--:--  0:00:06 --:--:-- 51.7M
Downloading https://drive.google.com/uc?export=download&id=1UcR-zVoMs7DH5dj3N1bswkiQTA4dmKF4 as weights/yolov3-spp-ultralytics.pt... Done (6.7s)
image 1/2 data/samples/bus.jpg: 512x384 3 persons, 1 buss, 1 ties, Done. (0.583s)
image 2/2 data/samples/zidane.jpg: 320x512 2 persons, 1 ties, Done. (0.538s)
Results saved to /mnt/storage/home/lm19073/yolov3/output
Done. (1.261s)
```

就这两张图片，BC4用时1.261秒，而相同的运算Colab上用时1.616秒，已经快了20%，所以在长时间的训练任务上，BC4应该更加占优势。不过一个无法避免的问题是，连接到服务器所需要的VPN最大时长也是12小时，所以该做的备份工作还是要做好，如checkpoint的储存，results的写入等。

## Issues

### Python版本问题

选择运行：

![image-20200715193247282](https://i.loli.net/2020/07/16/bD7SsRUzqOQIvNF.png)

按照[issue](https://github.com/bastibe/transplant/issues/37#issuecomment-311269734)的说法，BC4的Python 3.4.5版本是导致syntax error的原因，需要≥3.5的版本。采用`module avail`能发现系统中安装有Anaconda，Anaconda中包含多个不同版本的Python，3.7版本够用，就选择它load到module list中来，之后的python命令就自动调用的是3.7版本的Python了。操作如下：

```shell
[lm19073@bc4login2 ~]$ module avail

--------------------- /mnt/storage/easybuild/modules/local ---------------------
 ...
 languages/anaconda3/3.6.5
 languages/anaconda3/3.7
 ...
[lm19073@bc4login2 ~]$ module load languages/anaconda3/3.7
[lm19073@bc4login2 ~]$ module list

Currently Loaded Modules:
  1) languages/java/sdk-1.8.0.141   2) languages/anaconda3/3.7
[lm19073@bc4login2 ~]$ python --version
Python 3.7.4
```
### libstdc++问题

![image-20200716135338142](https://i.loli.net/2020/07/16/zfTYo9PZaeAwFjH.png)

1. 一种可能是libstdc++.so.6没有指向新版本而是6.0.19版本，需要手动解除软链接并创建新软链接，见[CXXABI_1.3.9 not included in libstdc++.so.6](https://github.com/ContinuumIO/anaconda-issues/issues/5191#)#5191。

   进入libstdc++所在目录`/usr/lib/`，检查长格式软链接，

   ```shell
   [lm19073@bc4login2 lib64]$ pwd
   /usr/lib64
   [lm19073@bc4login2 lib64]$ ls libstdc++.so.6 -l
   lrwxrwxrwx 1 root root 19 Mar 21  2017 libstdc++.so.6 -> libstdc++.so.6.0.19
   ```

   需要把这个删掉然后重定义指向下面的地址，不过需要管理员的权限...只要是不乏通过自己的权限解决的办法，<mark>不推荐</mark>。

   ```shell
   /mnt/storage/software/languages/anaconda/Anaconda3-2019-3.7/lib/libstdc++.so.6.0.26
   ```

2. 第二种方法是一类方法，就是看是导入什么package的时候出现的error，然后针对性地重装或者更新，见：[Link](https://github.com/widdowquinn/pyani/issues/96#issuecomment-337157574)。依然需要管理员权限，<mark>不推荐</mark>。

3. 第三种方法是最有针对性的，<mark>推荐</mark>.分析得知这个问题的原因是我们导入的Anaconda的Python模块并没有调用适配的自己的库文件，而是调用了系统的库文件，这两个目录下的`libstdc++`有什么区别呢？是版本的区别。

   ```shell
   [lm19073@bc4login3 yolov3]$ ls /mnt/storage/software/languages/anaconda/Anaconda3-2019-3.7/lib/libstdc++.so.6 -l
   lrwxrwxrwx 1 root root 19 Dec  5  2019 /mnt/storage/software/languages/anaconda/Anaconda3-2019-3.7/lib/libstdc++.so.6 -> libstdc++.so.6.0.26
   [lm19073@bc4login3 yolov3]$ ls /usr/lib64/libstdc++.so.6 -l
   lrwxrwxrwx 1 root root 19 Mar 21  2017 /usr/lib64/libstdc++.so.6 -> libstdc++.so.6.0.19
   ```

   解决办法[^lib]就是让Anaconda调用自己的lib：将Anaconda的lib目录写入动态链接库(DLL)环境变量：

   [^lib]:[matplotlib导入错误](https://blog.csdn.net/qq_36501182/article/details/102969174)

   ```shell
   export LD_LIBRARY_PATH=/mnt/storage/software/languages/anaconda/Anaconda3-2019-3.7/lib:$LD_LIBRARY_PATH
   ```

---

## References

[3 个相见恨晚的 Google Colaboratory 奇技淫巧！](https://zhuanlan.zhihu.com/p/56581879)

[苦逼学生党的Google Colab使用心得](https://zhuanlan.zhihu.com/p/54389036)

[最新超算 500 强名单：整体算力进入“千万亿次时代”](http://www.mittrchina.com/news/4009)

[High Performance Computing](https://www.bristol.ac.uk/acrc/high-performance-computing/)