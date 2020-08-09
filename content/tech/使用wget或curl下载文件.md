---
title: "使用wget或curl下载文件"
subtitle: ""
date: 2020-08-09T02:32:51+08:00
categories: [Tech, Configuration]
tags: [wget, curl, Google Drive]
slug: "download-shared-files-using-wget-or-curl"
toc: false
displayCopyright: true
dropCap: false
mermaid: false
comments: true
---

本文讲解如何通过wget或者curl命令下载Google Drive上的文件。出发点是我的一个曲线上传数据集的任务，如果我用VPN翻墙再通过校园网VPN连接上计算机集群，上传速度大概是100-200k/s（11小时），如果单纯使用校园网VPN连接，上传速度更低（约20k/s），然而直接上传Google Drive的时间只有1.5小时，所以我决定，先将数据集上传至谷歌开车，再通过操作计算机集群资源下载数据集。脚本[代码](https://github.com/JinhangZhu/project-diary/blob/master/scripts/get_epichands.sh)。

<!--more-->

## 分享链接

将文件设置为公开，当前版本的谷歌开车生成的分享链接的格式是：

```
https://drive.google.com/file/d/<fileid>/view
```

在后续过程中，除了`fileid`这个变量，还有用`filename`表示文件名。举个例子，对于文件`epichands_anchors.zip`，分享链接为：

```
https://drive.google.com/file/d/1MJZAlgRi1TjrUcqn4SuMNPccrZrxgyLD/view?usp=sharing
```

那么，

```shell
filename='epichands_anchors.zip'
fileid='1MJZAlgRi1TjrUcqn4SuMNPccrZrxgyLD'
```

## wget命令

- 小文件下载

  ```bash
  wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O ${filename}
  ```

- 大文件下载

  ```bash
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
  ```
  
- 示例：

  ```shell
  #!/bin/bash
  
  # cd scratch place
  cd scratch/
  
  # Download zip dataset from Google Drive
  filename='epichands_anchors.zip'
  fileid='1MJZAlgRi1TjrUcqn4SuMNPccrZrxgyLD'
  wget --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O ${filename}
  
  # Unzip
  unzip -q ${filename}
  rm ${filename}
  
  # cd out
  cd
  ```


## curl命令下载

- 小文件 < 40MB

  ```shell
  curl -L -o ${filename} "https://drive.google.com/uc?export=download&id=${fileid}"
  ```

- 大文件 > 40MB

  ```shell
  curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
  curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
  rm ./cookie
  ```

- 示例：

  ```shell
  #!/bin/bash
  
  # cd scratch place
  cd scratch/
  
  # Download zip dataset from Google Drive
  filename="coco2014labels.zip"
  fileid="1s6-CmF5_SElM28r52P1OUrCcuXZN-SFo"
  curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
  curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
  rm ./cookie
  
  # Unzip
  unzip -q ${filename}
  rm ${filename}
  
  # cd out
  cd
  ```


## References

- [使用wget命令下载Google drive上的文件](https://blog.csdn.net/Mao_Jonah/article/details/88372086)
- [Downloading Shared Files on Google Drive Using Curl](https://gist.github.com/tanaikech/f0f2d122e05bf5f971611258c22c110f)
- [get_coco2014.sh](https://github.com/JinhangZhu/yolov3/blob/custom/data/get_coco2014.sh)
- https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805#gistcomment-2316906
- [The Linux 'unzip' Command](https://www.lifewire.com/examples-linux-unzip-command-2201157)
- https://gist.github.com/vladalive/535cc2aff8a9527f1d9443b036320672