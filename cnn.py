# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 09:20:00 2018

@author: l
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#每个批次的大小
batch_size=100

#计算一共有多少批次
n_batch=mnist.train.num_examples//batch_size

#初始化权值
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    
#初始化偏置
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
#卷积层
def conv2d(x,W):
    #x input tensor of shape '[batch.in_height,in_width,in_channels]'
    #filter / kernel tensor of shape [filter_height,filter_width,in_channels,out_channels]
    #strides[0]=strides[3]=1'.stride[1]代表x方向的步长，stride[2],代表y方向的步长
    #padding:A 'string' from "SAME","VALID"
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME",use_cudnn_on_gpu=True)
    
#池化层
def max_pool_2x2(x):
    #ksize[1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    
#定义两个placeholder
x=tf.placeholder(tf.float32,[None,784])#28*28
y=tf.placeholder(tf.float32,[None,10])

#改变x的格式转换为4D的向量[batch,in_height,in_width,in_channels]
x_image=tf.reshape(x,[-1,28,28,1])

#初始化第一个卷积层的权重和偏置
W_conv1=weight_variable([5,5,1,32])#5*5采样窗口，32个卷积核从1个平面抽取特征
b_conv1=bias_variable([32])#每一个卷积核一个偏置值

#把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu函数进行激活
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)#max-pooling

#初始化第二个卷积层的权值和偏置
W_conv2=weight_variable([5,5,32,64])#5*5的采样窗口，64个卷积核从32个平面抽取特征
b_conv2=bias_variable([64])

#把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu函数进行激活
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)#max-pooling

#28*28的图片第一次卷积后还是28*28,第一次池化后变成14*14
#第二次卷积后为14*14,第二次池化后成为7*7
#通过上面操作后成为了64张7*7的平面

#初始化第一个全链接层的权值
W_fc1=weight_variable([7*7*64,1024])#上面操作得到64*7*7个神经元，全连接层有1024个神经元
b_fc1=bias_variable([1024])

#把池化层2的输出扁平为1维
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
#求第一个全连接层的输出
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#用keep_prob来表示神经元的输出概率
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#初始化第二个全链接层的权值
W_fc2=weight_variable([1024,10])#上面操作得到64*7*7个神经元，全连接层有1024个神经元
b_fc2=bias_variable([10])

#计算输出
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


#代价函数
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#学习
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#初始化变量
init=tf.global_variables_initializer()


#结果存放在布尔列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回最大值所在的位置
#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range (n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
            
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1})
        #train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1})        
        print("Iter:"+str(epoch)+"  Test_Accuracy:"+str(test_acc))
