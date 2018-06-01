import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y_ = tf.placeholder(tf.int32)

init = tf.contrib.layers.xavier_initializer()

conv1_w = tf.get_variable("conv1_w", shape = [5, 5, 3, 16], initializer = init)
conv2_w = tf.get_variable("conv2_w", shape = [5, 5, 16, 32], initializer = init)
fc1_w = tf.get_variable("fc1_w", shape = [8*8*32, 256], initializer = init)
fc2_w = tf.get_variable("fc2_w", shape = [256, 128], initializer = init)
logits_w = tf.get_variable("logits_w", shape = [128, 10], initializer = init)

conv1_b = tf.Variable(tf.zeros([16]))
conv2_b = tf.Variable(tf.zeros([32]))
fc1_b = tf.Variable(tf.zeros([256]))
fc2_b = tf.Variable(tf.zeros([128]))
logits_b = tf.Variable(tf.zeros([10]))

def model():
	input_layer = tf.reshape(X, [-1, 32, 32, 3])
	
	conv1 = tf.nn.conv2d(input_layer, conv1_w, strides = [1, 1, 1, 1], padding = "SAME")
	conv1 = tf.nn.relu(conv1 + conv1_b)
	
	pool1 = tf.nn.max_pool(conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = "SAME")
	
	conv2 = tf.nn.conv2d(pool1, conv2_w, strides = [1, 1, 1, 1], padding = "SAME")
	conv2 = tf.nn.relu(conv2 + conv2_b)
	
	pool2 = tf.nn.max_pool(conv2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = "SAME")
	
	pool2_flat = tf.reshape(pool2, [-1, 8*8*32])
	fc1 = tf.nn.relu( tf.matmul(pool2_flat, fc1_w) + fc1_b )
	fc2 = tf.nn.relu( tf.matmul(fc1, fc2_w) + fc2_b )
	
	logits = tf.matmul(fc2, logits_w) + logits_b
	return logits

