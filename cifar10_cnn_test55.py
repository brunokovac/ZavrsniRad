import tensorflow as tf
import numpy as np
import cifar10_cnn_model as cnn_model
	
def test(test_data, test_labels):
	logits = cnn_model.model()
	
	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, "drive/zavrsni/model")
		
		print("*"*25, "TEST", "*"*25)
		correct = tf.equal( tf.transpose(tf.cast(Y_, tf.int64)), tf.argmax(logits, 1) )
		accuracy = tf.reduce_mean(tf.cast(correct, "float"))
		print("Accuracy:", sess.run(accuracy, feed_dict = {X : test_data, Y_ : test_labels}))
		print("*"*55)
		

if __name__ == "__main__":
	cifar_train, cifar_test = tf.keras.datasets.cifar10.load_data()
	cifar_train = (cifar_train[0].astype('float32'), cifar_train[1])
	cifar_test = (cifar_test[0].astype('float32'), cifar_test[1])
		
	train_data = cifar_train[0]
	train_labels = np.asarray(cifar_train[1], dtype=np.int32)
	
	test_data = cifar_test[0]
	test_labels = np.asarray(cifar_test[1], dtype=np.int32)
	
	data_mean = train_data.mean((0, 1, 2))
	data_std = train_data.std((0, 1, 2))

	train_data = (train_data - data_mean) / data_std
	test_data = (test_data - data_mean) / data_std
	
	X = cnn_model.X
	Y_ = cnn_model.Y_
	test(test_data, test_labels)

