import tensorflow as tf
import numpy as np
import cifar10_cnn_model as cnn_model

def shuffle(data, labels):
	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)
	return data[indices], labels[indices]
	
def train(train_data, train_labels, eval_data, eval_labels):
	logits = cnn_model.model()
	loss = tf.losses.sparse_softmax_cross_entropy(labels = Y_, logits = logits)
	optimizer = tf.train.GradientDescentOptimizer(5e-3).minimize(loss)
	
	batch_size = 200
	num_data = train_data.shape[0]
	num_epochs = 200
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		best_accuracy = -1
		
		for epoch in range(num_epochs):
			epoch_loss = 0
			for i in range(num_data//batch_size):
				epoch_data = train_data[i*batch_size : i*batch_size + batch_size, :]
				epoch_labels = train_labels[i*batch_size : i*batch_size + batch_size, :]
				
				epoch_data, epoch_labels = shuffle(epoch_data, epoch_labels)
				
				_, loss1 = sess.run( [optimizer, loss], feed_dict = {X : epoch_data, Y_ : epoch_labels} )
				
			print("Epoch", epoch, ":", sess.run( loss, feed_dict = {X : train_data, Y_ : train_labels} ))
			correct = tf.equal( tf.transpose(tf.cast(Y_, tf.int64)), tf.argmax(logits, 1) )
			accuracy_tf = tf.reduce_mean(tf.cast(correct, "float"))
			accuracy = sess.run(accuracy_tf, feed_dict = {X : eval_data, Y_ : eval_labels})
			print("Accuracy:", accuracy)
			print("*"*50)
			
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				saver = tf.train.Saver()
				saver.save(sess, "drive/zavrsni/model")
		

if __name__ == "__main__":
	import sys
	args = sys.argv
	
	EVAL_SIZE = 0
	if len(args) > 1 and int(args[1]) > 0:
		EVAL_SIZE = int(args[1])
		print("Training with validation set of size", EVAL_SIZE)
	else:
		sys.exit(0)

	cifar_train, cifar_test = tf.keras.datasets.cifar10.load_data()
	cifar_train = (cifar_train[0].astype('float32'), cifar_train[1])
	cifar_test = (cifar_test[0].astype('float32'), cifar_test[1])
		
	train_data = cifar_train[0][EVAL_SIZE:] / 255
	train_labels = np.asarray(cifar_train[1][EVAL_SIZE:], dtype=np.int32)
	
	eval_data = cifar_train[0][0:EVAL_SIZE] / 255
	eval_labels = np.asarray(cifar_train[1][0:EVAL_SIZE], dtype=np.int32)
	
	test_data = cifar_test[0] / 255
	test_labels = np.asarray(cifar_test[1], dtype=np.int32)
	
	data_mean = train_data.mean((0, 1, 2))
	data_std = train_data.std((0, 1, 2))

	train_data = (train_data - data_mean) / data_std
	eval_data = (eval_data - data_mean) / data_std
	test_data = (test_data - data_mean) / data_std
	
	X = cnn_model.X
	Y_ = cnn_model.Y_
	train(train_data, train_labels, eval_data, eval_labels)

