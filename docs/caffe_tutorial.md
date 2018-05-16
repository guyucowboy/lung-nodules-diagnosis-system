# Caffe Tutorial

by Wentai Zhang

## Tutorial Collection

(feel free to add yours)

1. http://suanfazu.com/t/ubuntu-14-04-caffe/447

   The navigation on the left has more articles following.

2. http://blog.csdn.net/hjimce/article/category/3163421

   A big collection. Very detailed and solid.

3. http://www.cnblogs.com/denny402/category/759199.html

   Well-explained caffe tutorial. Good for beginners.

## Step by Step

In my opinion, the best way to learn a program is to #copy# it.
In this copying process, you know how to write and build your neural networks.

0. Learn how to install Caffe

   Many tutorials. In the above collections, you can find what you want.

1. MNIST dataset experience

   You can follow this guide: [Training LeNet on MNIST with Caffe](http://caffe.berkeleyvision.org/gathered/examples/mnist.html).

   Or any examples from "Examples" in [Caffe](http://caffe.berkeleyvision.org/).

2. Debug

   Use a debug tool, and trace all the way through. By this method, you will know how Caffe works.
   
   You should step over the top functions, such as Solver, Net, Layer, Specific Layer.

3. Write your own program

   For now, you have reviewed several programs written by others.
   
   You should try to modify and test them. 

4. Complex funtions

   It is a little bit difficult to implement some functions (e.g. backpropagation) in Caffe.
   
   Right now, you may refer to some advanced tutorials written by someone who is very familiar with Caffe.
   
   * [UFLDL Tutorial](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial)
   * [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)
   * [Hacker's guide to Neural Networks](http://karpathy.github.io/neuralnets/)
   * [Making a Caffe Layer](https://chrischoy.github.io/research/making-caffe-layer/)

For every step, you many find many helpful documents on the Internet.
Make sure you find the latest version because Caffe updates frequently.
