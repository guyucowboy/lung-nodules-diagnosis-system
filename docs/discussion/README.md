# Discussion

This folder stores the weekly discussion slides.
This file records the progression of our project.

# Weekly Milestones

## Week 7~8

1. Successfully rewrite our codes using TensorFlow.
We now have incorrect inaccuracy (~100%).
A visual debugger will be helpful.
2. Next step is to find out why this anomaly happens.

## Week 6

1. Successfully extract the input data from raw test sets. (of size 32^3)
2. Recommend segmentation and downsampling to obtain high quality data. (maybe another NN)

## Week 4~5

1. Exam Weeks.
2. Input data are ready. Initial caffe framework is done.
Start writing source codes, and produce the results.
During this process, we can then understand the mechanics behind these neural networks.

TODO:
1. Segmentation: foreground and background
2. Caffe GPU-enable
3. Caffe tutorials

## Week 3

1. Divide the image into 45 * 45 * 45 blocks
However, these smaller blocks seem a little blurred.
2. LUNA'16 report of Prof. Wang.
3. Ted Liu could upload some explanatory documents of current Caffe framework.

## Week 1~2

1. Taide Liu & Haochen Li: Almost finished a Caffe framework; waited for input data.
2. Zekun Li & Pengfei Dou: Reorganizing data into LMDB format.
3. Yuchen Zhou: Preparing final exams.
4. Hanxian Huang: Reviewing Medical NN.

We may need some more discussions on how to determine the inputs and features.

# Daily Progress

## 20170520

* Create the repo.
* Upload data to the server.
* Setup runtime environment.

## 20170521

* 搜集天池讨论圈中的图像预处理帖子，配置环境

## 20170522

* 根据天池讨论圈中的内容搜寻到Github上DSB2017比赛的一些教程

## 20170523

* 根据前两天搜集的资料实现了一些预处理的方式，并获得了一小部分处理结果

## 20170524

* 继续尝试预处理方式，并且有比较好的进展。关注到**天池官方更正了一些已被发现错误的数据**，并发布了改正后的数据。

## 20170525

* 采用Github上DSB2017比赛提供的可视化处理方法取得成功，将**原来的mhd数据处理为3切片的png图片**。在服务器上调试运行成功，代码和数据在服务器上test文件夹里，图片输出在test/tutorial里。
